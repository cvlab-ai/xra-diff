
from reconsnet.train import train_with_clearml
from reconsnet.model.gan import GANModule
from reconsnet.data.dataset import get_dm_both


train_with_clearml(
    "gan_both",
    GANModule,
    get_dm_both("/home/shared/imagecas/projections_split/train", "/home/shared/imagecas/projections_split/val")
)
