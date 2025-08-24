import torch

from dataclasses import dataclass

@dataclass
class XRay:
    img: torch.Tensor
    sid: torch.FloatType
    sod: torch.FloatType
    alpha: torch.FloatType # radians
    beta: torch.FloatType # radians
    spacing: torch.FloatType
    size: tuple[torch.IntType, torch.IntType]
