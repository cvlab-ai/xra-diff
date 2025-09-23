'''
test on a synthesized dataset
'''
import torch.nn.functional as F

from reconsnet.util.test import synthetic_test, synthetic_test_adaptive
from reconsnet.model.unet3d import Unet3DModule
from reconsnet.data.dataset import XRayDatasetRight

from torch.utils.data import Subset

CHECKPOINT_PATH = "unet3d-baseline.ckpt"
DATA_PATH = "/home/shared/imagecas/projections_split/val"
RESULTS_PATH = "data/synthetic_adaptive_unet_right.csv"
MODEL = Unet3DModule.load_from_checkpoint(CHECKPOINT_PATH, lr=1e-5)
RECONSTRUCT = lambda x: MODEL.forward(x[0])

synthetic_test_adaptive(
    model=MODEL,
    ds=XRayDatasetRight(root_dir=DATA_PATH),
    csv_output_path=RESULTS_PATH,
    reconstruct=RECONSTRUCT,
    repeat_each=1
)
