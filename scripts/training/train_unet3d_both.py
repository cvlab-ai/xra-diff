from reconsnet.train import train_with_clearml
from reconsnet.model.unet3d import Unet3DModule
from reconsnet.data.dataset import get_dm_both


train_with_clearml(
    "unet3d_both",
    Unet3DModule,
    get_dm_both("/home/shared/imagecas/projections_split/train", "/home/shared/imagecas/projections_split/val")
)