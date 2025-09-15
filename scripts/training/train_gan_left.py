
from reconsnet.train import train_with_clearml
from reconsnet.model.gan import GANModule
from reconsnet.data.dataset import get_dm_left


train_with_clearml(
    "gan_left",
    GANModule,
    get_dm_left("/home/shared/imagecas/projections_split/train", "/home/shared/imagecas/projections_split/val")
)
