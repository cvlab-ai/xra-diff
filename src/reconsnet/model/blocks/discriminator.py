import torch
import torch.nn as nn

from .conv import ConvBlock
from .dsconv import DSConv


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.ds_conv = DSConv(1, 48)
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
