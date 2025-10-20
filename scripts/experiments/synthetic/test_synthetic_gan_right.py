from reconsnet.util.test import synthetic_test
from reconsnet.model.gan import GANModule
from reconsnet.data.dataset import XRayDatasetRight

CHECKPOINT_PATH = "/home/shared/model-weights/baseline-gan-right.ckpt"
DATA_PATH = "/home/shared/imagecas/projections_split/val"
RESULTS_PATH = "data/synthetic_gan_right.csv"
MODEL = GANModule.load_from_checkpoint(CHECKPOINT_PATH)
RECONSTRUCT = lambda x: MODEL.generator.forward(x[0])

synthetic_test(
    model=MODEL,
    ds=XRayDatasetRight(root_dir=DATA_PATH),
    csv_output_path=RESULTS_PATH,
    reconstruct=RECONSTRUCT,
    repeat_each=1
)
