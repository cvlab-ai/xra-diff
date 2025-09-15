
from reconsnet.train import train_with_clearml
from reconsnet.model.diffusion import DiffusionModule
from reconsnet.data.dataset import get_dm_right


train_with_clearml(
    "diffusion_right",
    DiffusionModule,
    get_dm_right("/home/shared/imagecas/projections_split/train", "/home/shared/imagecas/projections_split/val")
)
