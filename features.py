import math
import pywt
import numpy as np
import torch

from torch import nn


class FourierFeatures(nn.Module):
    def __init__(self, in_features=3, out_features=16, std=1.):
        super().__init__()
        assert out_features % 2 == 0  # Ensure the out_features is even for cos and sin pairs
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features // 2, in_features) * std, requires_grad=False)  # non-learnable

    def forward(self, input):
        # Reshape input to (H*W, C)
        C, H, W = input.shape
        input_reshaped = input.view(H*W, C)

        # Apply Fourier transform
        f = 2 * math.pi * input_reshaped @ self.weight.T
        
        # Reshape f to (B, H, W, out_features)
        fourier_features = torch.cat([f.cos(), f.sin()], dim=-1).view(H, W, self.out_features)

        # Transpose to get (out_features, H, W)
        return fourier_features.permute(2, 0, 1)
