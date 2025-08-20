import torch

from dataclasses import dataclass


@dataclass
class ConeBeamView:
    src: torch.Tensor
    sid: torch.FloatType
    sod: torch.FloatType
    
    # image - detector plane definition
    i_origin: torch.Tensor
    i_u: torch.Tensor # basis along width
    i_v: torch.Tensor # basis along height
    i_spacing: tuple[torch.IntType, torch.IntType]
    

@dataclass
class StereoView:
    view0: ConeBeamView
    view1: ConeBeamView
