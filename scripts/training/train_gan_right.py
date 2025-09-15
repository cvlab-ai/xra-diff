
from reconsnet.train import train_with_clearml
from reconsnet.model.gan import GANModule
from reconsnet.data.dataset import get_dm_right


train_with_clearml(
    "gan_right",
    GANModule,
    get_dm_right("/home/shared/imagecas/projections_split/train", "/home/shared/imagecas/projections_split/val")
)
