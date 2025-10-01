from reconsnet.train import train_with_clearml
from reconsnet.model.diffusion import DiffusionModule
from reconsnet.data.dataset import get_dm_both


train_with_clearml(
    "diffusion_both",
    DiffusionModule,
    get_dm_both("/home/shared/imagecas/projections_split/train", "/home/shared/imagecas/projections_split/val")
)
