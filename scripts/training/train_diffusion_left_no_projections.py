from reconsnet.train import train_with_clearml
from reconsnet.model.diffusion import DiffusionModule
from reconsnet.data.dataset import get_dm_left

from functools import partial

train_with_clearml(
    "diffusion_left_no_projections",
    partial(DiffusionModule, take_projections=False),
    get_dm_left("/home/shared/imagecas/projections_split/train", "/home/shared/imagecas/projections_split/val")
)
