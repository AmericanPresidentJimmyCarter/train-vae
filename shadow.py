import gc

import torch

class ShadowModel:
    def __init__(self):
        self.shadow_weights = None

    def store(self, model):
        """
        Store the model parameters and their values.

        Args:
            model (torch.nn.Module): Model whose parameters you want to store.
        """
        # Ensure that the model is on the CPU and store its state_dict (weights)
        self.shadow_weights = {name: param.data.detach().clone().cpu() 
                               for name, param in model.state_dict().items()}

    def restore(self, model):
        """
        Restore the stored parameters to the given model.

        Args:
            model (torch.nn.Module): Model to which you want to restore the stored parameters.
        """
        if self.shadow_weights is None:
            raise ValueError("No weights have been stored in the ShadowModel. Call the 'store' method first.")

        gc.collect()
        torch.cuda.empty_cache()

        # Load the stored weights to the model
        with torch.no_grad():
            for (cpu_name, cpu_param), (gpu_name, gpu_param) in zip(
                    self.shadow_weights.items(),
                    model.named_parameters(),
                ):
                assert cpu_name == gpu_name
                gpu_param.copy_(cpu_param)

        gc.collect()
        torch.cuda.empty_cache()
