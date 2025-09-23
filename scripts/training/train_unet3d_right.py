
from reconsnet.train import train_with_clearml
from reconsnet.model.unet3d import Unet3D
from reconsnet.data.dataset import get_dm_right


train_with_clearml(
    "unet3d_right",
    Unet3D,
    get_dm_right("/home/shared/imagecas/projections_split/train", "/home/shared/imagecas/projections_split/val")
)
