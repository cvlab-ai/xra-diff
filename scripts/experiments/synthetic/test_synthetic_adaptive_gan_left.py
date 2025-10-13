from reconsnet.util.test import synthetic_test_adaptive
from reconsnet.model.gan import GANModule
from reconsnet.data.dataset import XRayDatasetLeft

CHECKPOINT_PATH = "/home/shared/model-weights/baseline-gan-left.ckpt"
DATA_PATH = "/home/shared/imagecas/projections_split/pilot"
RESULTS_PATH = "data/synthetic_adaptive_gan_left.csv"
MODEL = GANModule.load_from_checkpoint(CHECKPOINT_PATH)
RECONSTRUCT = lambda x: MODEL.generator.forward(x[0])

synthetic_test_adaptive(
    model=MODEL,
    ds=XRayDatasetLeft(root_dir=DATA_PATH),
    csv_output_path=RESULTS_PATH,
    reconstruct=RECONSTRUCT,
    repeat_each=1,
)
