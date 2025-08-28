import torch
import torch.nn as nn

from diffusers import UNet3DConditionModel

from ..config import get_config


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=4, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2, 1),
        )
        
    def forward(self, x):
        return self.conv(x)
    

class DSConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        # Temporary
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x):
        return self.conv(x)


class ConditionalGenerator(nn.Module):
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
    
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.ds_conv = DSConv()
        self.conv1 = ConvBlock(1, 64)
        self.conv2 = ConvBlock(64, 128)
        self.conv3 = ConvBlock(128, 256)
        self.conv4 =  nn.Conv3d(256, 1, kernel_size=1, stride=1, padding=1, bias=False),
        self.tanh = nn.Tanh()
    
    def forward(self, x1, x2):
        x1 = self.conv1(x1)
        x2 = self.ds_conv(x2)
        concated = torch.concat(x1, x2)
        x3 = self.conv2(concated)
        x4 = self.conv3(x3)
        x5 = self.conv4(x4)
        out = torch.mean(self.tanh(x5))
        
        return out 

