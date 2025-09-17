'''
test on a synthesized dataset
'''
import torch.nn.functional as F

from reconsnet.util.test import synthetic_test
from reconsnet.model.gan import GANModule
from reconsnet.data.dataset import XRayDatasetRight

from torch.utils.data import Subset

CHECKPOINT_PATH = "baseline.ckpt"
DATA_PATH = "/home/shared/imagecas/projections_split/val"
RESULTS_PATH = "data/synthetic_baseline_right.csv"
MODEL = GANModule.load_from_checkpoint(CHECKPOINT_PATH)
RECONSTRUCT = lambda x: F.sigmoid(MODEL.generator.forward(x[0]))

synthetic_test(
    model=MODEL,
    ds=XRayDatasetRight(root_dir=DATA_PATH),
    csv_output_path=RESULTS_PATH,
    reconstruct=RECONSTRUCT,
    repeat_each=1
)
