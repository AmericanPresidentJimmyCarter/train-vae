"""
TODO: fix training mixed precision -- issue with AdamW optimizer
"""

import argparse
import gc
import logging
import math
import os
import random

# import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision

from PIL import Image
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from diffusers.training_utils import EMAModel
from torchvision import transforms
from tqdm.auto import tqdm

from diffusers import AutoencoderKL
from diffusers.utils import is_wandb_available

import lpips

from features import FourierFeatures
from discriminator import NLayerDiscriminator, hinge_d_loss, vanilla_g_loss
from shadow import ShadowModel

if is_wandb_available():
    import wandb

logger = get_logger(__name__, log_level="INFO")


transform_from_pil = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])
transform_to_pil = transforms.ToPILImage()


train_transforms = lambda resolution: transforms.Compose(
    [
        # transforms.Resize(
        #     args.resolution, interpolation=transforms.InterpolationMode.BILINEAR
        # ),
        transforms.RandomCrop(resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)
ff_featurizer = FourierFeatures()


def train_feature_extract(img, resolution, return_img=False):
    img = resize_to_min_size_pil(img, resolution)

    if return_img:
        img = random_crop_to_max_size_pil(img, resolution)
        img_tens = transform_from_pil(img)
    else:
        img_tens = train_transforms(resolution)(img)
    ff_features = ff_featurizer(img_tens)

    if return_img:
        return torch.cat([img_tens, ff_features], dim=0), img
    return torch.cat([img_tens, ff_features], dim=0)


def encode_img(vae, input_img, max_size=512, is_pil=False):
    output_img = None
    if is_pil:
        input_img, output_img = train_feature_extract(input_img, max_size, return_img=True)
        input_img = input_img.to('cuda', dtype=torch.bfloat16)
    if len(input_img.shape)<19:
        input_img = input_img.unsqueeze(0)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        latent = vae.encode(input_img) # Note scaling
    return 0.18215 * latent.latent_dist.sample(), output_img


def decode_img(vae, latents, return_pil=False):
    # bath of latents -> list of images
    latents = (1 / 0.18215) * latents
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        image = vae.decode(latents.to('cuda')).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach()
    if return_pil:
        transform_to_pil(image.squeeze(0))
    return image


def round_down_and_crop(tensor):
    b, c, h, w = tensor.size()

    # Compute the target dimensions by rounding down to nearest 8
    target_h = (h // 8) * 8
    target_w = (w // 8) * 8

    # Crop the tensor
    tensor_cropped = tensor[:, :, :target_h, :target_w]

    return tensor_cropped


def patch_val_loader(loader):
        ori_begin_method = loader.begin
        ori_end_method = loader.end
        loader.begin = lambda: (ori_begin_method(), ori_end_method())[0]
        loader.end = lambda: None


@torch.no_grad()
def log_validation(test_dataloader, vae, accelerator, weight_dtype, epoch):
    logger.info("Running validation... ")
    vae_model = vae

    with torch.no_grad():
        images = []
        test_dataloader_iter = iter(test_dataloader)

        img_1 = Image.open('test1.jpg')
        img_1_encoded, img_1 = encode_img(vae_model, img_1, is_pil=True)
        img_1_recon = decode_img(vae_model, img_1_encoded)
        images.append(
            torch.cat([transform_from_pil(img_1).unsqueeze(0).cpu(), img_1_recon.cpu()], axis=0)
        )

        img_2 = Image.open('test2.jpg')
        img_2_encoded, img_2 = encode_img(vae_model, img_2, is_pil=True)
        img_2_recon = decode_img(vae_model, img_2_encoded)
        images.append(
            torch.cat([transform_from_pil(img_2).unsqueeze(0).cpu(), img_2_recon.cpu()], axis=0)
        )

        for _ in enumerate(range(4)):
            x = next(test_dataloader_iter)[0]['image']
            # x = x.squeeze(0)
            # x = transform_to_pil(x)
            # reconstructions = vae_model(x).sample
            img_encoded, img_crop = encode_img(vae_model, x, is_pil=True)
            reconstructions = decode_img(vae_model, img_encoded)

            images.append(
                torch.cat([transform_from_pil(img_crop).unsqueeze(0).cpu(), reconstructions.cpu()], axis=0)
            )

        for tracker in accelerator.trackers:
            if tracker.name == "tensorboard":
                np_images = np.stack([np.asarray(img) for img in images])
                tracker.writer.add_images(
                    "Original (left) / Reconstruction (right)", np_images, epoch
                )
            elif tracker.name == "wandb":
                tracker.log(
                    {
                        "Original (left) / Reconstruction (right)": [
                            wandb.Image(torchvision.utils.make_grid(image))
                            for _, image in enumerate(images)
                        ]
                    }
                )
            else:
                logger.warn(f"image logging not implemented for {tracker.gen_images}")

    del vae_model
    torch.cuda.empty_cache()


def random_crop_to_max_size(tensor, target_size=512):
    b, c, h, w = tensor.size()

    # If height is greater than target_size, select a random start index
    if h > target_size:
        top = torch.randint(0, h - target_size + 1, (1,)).item()
    else:
        top = 0

    # If width is greater than target_size, select a random start index
    if w > target_size:
        left = torch.randint(0, w - target_size + 1, (1,)).item()
    else:
        left = 0

    # Determine end indices based on the start indices
    bottom = min(h, top + target_size)
    right = min(w, left + target_size)

    # Perform the crop
    tensor_cropped = tensor[:, :, top:bottom, left:right]

    return tensor_cropped


def random_crop_to_max_size_pil(img, target_size=256):
    # Make sure the max crop size is not larger than the image
    crop_size = target_size

    # Generate a random position for the crop box
    left = random.randint(0, img.size[0] - crop_size)
    top = random.randint(0, img.size[1] - crop_size)

    # The crop rectangle should be square
    right = left + crop_size
    bottom = top + crop_size

    # Crop the image
    cropped_image = img.crop((left, top, right, bottom))

    return cropped_image


def resize_to_min_size_pil(image, min_size):
    # Get the original size of the image (width, height)
    original_size = image.size
    
    # Determine the smaller edge
    min_edge_index = original_size.index(min(original_size))
    
    # Check if the smaller edge is less than the minimum size
    if original_size[min_edge_index] < min_size:
        # Calculate the scale factor needed to upscale the smaller edge to min_size
        scale_factor = min_size / original_size[min_edge_index]
        
        # Compute the new size while maintaining aspect ratio
        new_size = [round(scale_factor * dim) for dim in original_size]
        
        # Resize the image
        image = image.resize(new_size, Image.LANCZOS)
    
    return image


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simple example of a VAE training script."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_ema_model_name_or_path",
        type=str,
        default=None,
        required=False,
        help="Path to pretrained EMA model or model identifier",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--use_ema", action="store_true", help="Whether or not to use EMA model."
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--shadow_model_every",
        type=int,
        default=0,
        help="Keeps a copy of the model on CPU to restore from if the loss reaches NaN every n steps.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        '--dataloader_num_workers',
        type=int,
        default=4,
    )
    parser.add_argument(
        "--image_column",
        type=str,
        default="image",
        help="The column of the dataset containing an image.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="vae-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--use_discriminator",
        action="store_true",
        help="Whether or not to use a discriminator for loss.",
    )
    parser.add_argument(
        "--discriminator_scale",
        type=float,
        default=1e-3,
        help="Loss scaling for the discriminator.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--discriminator_warmup_steps",
        type=int,
        default=100,
        help="Number of steps for the warmup for discriminator.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs_vae",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Checkpoint path to resume from"
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=1000,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=(
            "Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator` `ProjectConfiguration`."
            " See Accelerator::save_state https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.save_state"
            " for more docs"
        ),
    )
    parser.add_argument(
        "--test_samples",
        type=int,
        default=4,
        help="Number of images to remove from training set to be used as validation.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=500,
        help="Run validation every X steps.",
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="vae-testing",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument(
        "--kl_scale",
        type=float,
        default=1e-6,
        help="Scaling factor for the Kullback-Leibler divergence penalty term.",
    )
    parser.add_argument(
        "--lpips_scale",
        type=float,
        default=1e-1,
        help="Scaling factor for the LPIPS metric",
    )

    args = parser.parse_args()

    # Sanity checks
    # if args.dataset_name is None and args.train_data_dir is None:
    #     raise ValueError("Need either a dataset name or a training folder.")

    return args


def main():
    args = parse_args()

    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(
        total_limit=args.checkpoints_total_limit,
        project_dir=args.output_dir,
        logging_dir=logging_dir,
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    # if args.seed is not None:
    #     set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load vae
    vae = None
    ema_vae = None
    if args.pretrained_model_name_or_path:
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path, revision=args.revision
        )
    else:
        vae = AutoencoderKL.from_config('./config.json')

    if args.use_ema and args.pretrained_ema_model_name_or_path:
        ema_vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path, revision=args.revision
        )
        ema_vae = EMAModel(ema_vae.parameters(), model_cls=AutoencoderKL, model_config=ema_vae.config)
    if args.use_ema and not args.pretrained_ema_model_name_or_path:
        ema_vae = AutoencoderKL.from_config('./config.json')
        ema_vae_sd = ema_vae.state_dict()
        for name, param in vae.named_parameters():
            ema_vae_sd[name].copy_(param)
        ema_vae = EMAModel(ema_vae.parameters(), model_cls=AutoencoderKL, model_config=ema_vae.config)

    # vae = torch.compile(vae, dynamic=True)

    vae_params = vae.parameters()

    if args.gradient_checkpointing:
        vae.enable_gradient_checkpointing()

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    optimizer = optimizer_class(vae_params, lr=args.learning_rate)

    accelerator_d = None
    discriminator = None
    optimizer_d = None
    if args.use_discriminator:
        accelerator_d = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision=args.mixed_precision,
            log_with=args.report_to,
            project_config=accelerator_project_config,
        )
        discriminator = NLayerDiscriminator()
        optimizer_d = optimizer_class(discriminator.parameters(), lr=1e-3, betas=(0.9, 0.999))

        (
            discriminator,
            optimizer_d,
            _,
            _,
            _,
        ) = accelerator_d.prepare(
            discriminator, optimizer_d, None, None, None
        )


    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
        )
    else:
        data_files = {}
        if args.train_data_dir is not None:
            data_files["train"] = os.path.join(args.train_data_dir, "**")
        dataset = load_dataset(
            "imagefolder",
            data_files=data_files,
            cache_dir=args.cache_dir,
        )

    column_names = dataset["train"].column_names
    if args.image_column is None:
        image_column = column_names[0]
    else:
        image_column = args.image_column
        if image_column not in column_names:
            raise ValueError(
                f"--image_column' value '{args.image_column}' needs to be one of: {', '.join(column_names)}"
            )

    def preprocess(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        examples["pixel_values"] = [train_feature_extract(image, args.resolution) for image in images]
        return examples

    with accelerator.main_process_first():
        # Split into train/test
        dataset = dataset["train"].train_test_split(test_size=args.test_samples)
        # Set the training transforms
        train_dataset = dataset["train"].with_transform(preprocess)
        test_dataset = dataset["test"]

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        return {"pixel_values": pixel_values}

    def worker_init_fn(worker_id):
        """Set the random seed based on worker id."""
        # You can use any seed generation strategy here
        seed = (args.seed or torch.initial_seed()) + worker_id
        random.seed(seed)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        worker_init_fn=worker_init_fn,
        collate_fn=collate_fn,
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, shuffle=True, batch_size=1, collate_fn=lambda x: x
    )

    # lr_scheduler = get_scheduler(
    #     args.lr_scheduler,
    #     optimizer=optimizer,
    #     num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
    #     num_training_steps=args.num_train_epochs * args.gradient_accumulation_steps,
    # )
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        total_steps=num_update_steps_per_epoch * args.num_train_epochs,
        max_lr=args.learning_rate,
        pct_start=0.,# 0.075,
        div_factor=3,
        final_div_factor=1,
    )

    # Prepare everything with our `accelerator`.
    (
        vae,
        optimizer,
        train_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        vae, optimizer, train_dataloader, lr_scheduler
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=weight_dtype)
    if args.use_discriminator:
        discriminator.to(accelerator.device, dtype=weight_dtype)
    if args.use_ema:
        ema_vae.to(accelerator.device)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, tracker_config)

    args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    # ------------------------------ TRAIN ------------------------------ #
    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num test samples = {len(test_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    if args.shadow_model_every > 0:
        logger.info(f"  Shadowing model against NaN every {args.shadow_model_every} steps")

    global_step = 0
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            if args.uses_accelerator:
                accelerator.load_state(os.path.join(args.output_dir, path, 'discriminator'))
            global_step = int(path.split("-")[1])
            first_epoch = global_step // num_update_steps_per_epoch

    first_epoch = 0

    progress_bar = tqdm(
        range(global_step, args.max_train_steps),
        initial=global_step,
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")

    lpips_loss_fn = lpips.LPIPS(net="alex").to(accelerator.device)

    shadow_model = None
    if args.shadow_model_every > 0:
        shadow_model = ShadowModel()
        shadow_model.store(accelerator.unwrap_model(vae))

    mini_steps = 0

    for epoch in range(first_epoch, args.num_train_epochs):
        vae.train()
        train_loss = 0.0
        last_failed = False
        for step, batch in enumerate(train_dataloader):
            target = batch['pixel_values'].to(weight_dtype)

            if args.use_discriminator:
                with torch.no_grad():
                    posterior = vae.encode(target).latent_dist
                    z = posterior.mode()
                    rec = vae.decode(z).sample
                with accelerator_d.accumulate(discriminator):
                    real_pred, fake_pred = discriminator(target[:, :3, :, :]), discriminator(rec.detach())
                    d_loss = hinge_d_loss(real_pred, fake_pred)
                    accelerator_d.backward(d_loss)
                    optimizer_d.step()
                    optimizer_d.zero_grad()
                with torch.no_grad():
                    fake_pred = discriminator(rec.detach())
                    g_loss = vanilla_g_loss(fake_pred)

            mini_steps += 1

            with accelerator.accumulate(vae):
                if last_failed:
                    last_failed = False
                    gc.collect()
                    torch.cuda.empty_cache()

                # https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/autoencoder_kl.py
                posterior = vae.encode(target).latent_dist
                z = posterior.mode()
                pred = vae.decode(z).sample

                kl_loss = posterior.kl().mean()
                mse_loss = F.mse_loss(pred, target[:, :3, :, :], reduction="mean")
                lpips_loss = lpips_loss_fn(pred, target[:, :3, :, :]).mean()

                loss = None
                if not args.use_discriminator:
                    loss = (
                        mse_loss + args.lpips_scale * lpips_loss + args.kl_scale * kl_loss
                    )
                else:
                    d_scale = args.discriminator_scale if \
                        global_step > args.discriminator_warmup_steps else \
                        0
                    loss = (
                        mse_loss +
                        (args.lpips_scale * lpips_loss) +
                        (d_scale * g_loss) +
                        (args.kl_scale * kl_loss)
                    )

                if shadow_model is not None and torch.isnan(loss).any():
                    if accelerator.is_main_process:
                        logger.info('NaN loss/weight collapse detected, restoring weights')
                    del pred, posterior, batch, target, loss, z, kl_loss, mse_loss, lpips_loss
                    optimizer.zero_grad()
                    gc.collect()
                    torch.cuda.empty_cache()
                    shadow_model.restore(accelerator.unwrap_model(vae))
                    last_failed = True
                    continue
                else:
                    # Gather the losses across all processes for logging (if we use distributed training).
                    avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                    train_loss += avg_loss.item() / args.gradient_accumulation_steps

                    params_to_clip = vae.decoder.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                    accelerator.backward(loss)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_vae.step(vae.parameters())

                logs = {
                    "step_loss": loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                    "mse": mse_loss.detach().item(),
                    "lpips": lpips_loss.detach().item(),
                    "kl": kl_loss.detach().item(),
                    "train_loss": train_loss,
                }
                if args.use_discriminator:
                    logs['d_loss'] = d_loss.detach().item()
                    logs['g_loss'] = g_loss.detach().item()
                accelerator.log(logs, step=global_step)
                progress_bar.set_postfix(**logs)

                progress_bar.update(1)
                global_step += 1
                train_loss = 0.0

                if accelerator.is_main_process:
                    if shadow_model is not None and \
                        global_step % args.shadow_model_every == 0:
                        shadow_model.store(accelerator.unwrap_model(vae))

                    if global_step % args.validation_steps == 0:
                        try:
                            if args.use_ema:
                                ema_vae_temp = AutoencoderKL.from_config('./config.json')
                                ema_vae.copy_to(ema_vae_temp.parameters())
                                ema_vae_temp.to(accelerator.device)
                                # Switch back to the original UNet parameters.
                                log_validation(test_dataloader, ema_vae_temp,
                                    accelerator, weight_dtype, epoch)
                                del ema_vae_temp
                                gc.collect()
                                torch.cuda.empty_cache()
                            else:
                                log_validation(test_dataloader,
                                    accelerator.unwrap_model(vae), accelerator,
                                    weight_dtype, epoch)
                        except RuntimeError as e:
                            logger.warn(f"Unable to run validation: {str(e)}")

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(
                            args.output_dir, f"checkpoint-{global_step}"
                        )
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                        save_path_vae = os.path.join(
                            args.output_dir, f"checkpoint-{global_step}", 'vae'
                        )
                        vae.save_pretrained(save_path_vae)

                        save_path_ema_vae = os.path.join(
                            args.output_dir, f"checkpoint-{global_step}", 'ema_vae'
                        )
                        ema_vae.save_pretrained(save_path_ema_vae)

                        if args.use_discriminator:
                            save_path_d = os.path.join(
                                args.output_dir, f"checkpoint-{global_step}", 'discriminator'
                            )
                            accelerator_d.save_state(save_path_d)

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        vae = accelerator.unwrap_model(vae)
        vae.save_pretrained(args.output_dir)
        if args.use_discriminator:
            discriminator = accelerator_d.unwrap_model(discriminator)
            discriminator.save_pretrained(os.path.join(args.output_dir, 'final_discriminator'))
        if args.use_ema:
            save_path_ema_vae = os.path.join(
                args.output_dir, 'final_ema_vae'
            )
            ema_vae.save_pretrained(save_path_ema_vae)

    accelerator.end_training()


if __name__ == "__main__":
    main()
