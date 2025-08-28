import torch
import torch.nn as nn

from diffusers import UNet3DConditionModel

from ...config import get_config


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        config = get_config()

        self.unet_3d = UNet3DConditionModel(
            sample_size=config['data']['grid_dim'],
            in_channels=1, # why 2?
            out_channels=1,
            **config['unet3d']
        )

    def forward(self, x):
        return self.unet_3d(x)