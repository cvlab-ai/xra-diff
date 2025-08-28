'''
test on a synthesized dataset
'''
from reconsnet.util.test import synthetic_test
from reconsnet.model.diffusion import DiffusionModule
from reconsnet.data.dataset import XRayDatasetRight

from torch.utils.data import Subset


CHECKPOINT_PATH = "vanilla-200.ckpt"
DATA_PATH = "/home/shared/imagecas/projections"
RESULTS_PATH = "data/synthetic_right.csv"
RESULTS_FAST_PATH = "data/synthetic_fast_right.csv"

synthetic_test(
    DiffusionModule.load_from_checkpoint(CHECKPOINT_PATH, lr=1e-4),
    ds = Subset(XRayDatasetRight(root_dir=DATA_PATH), [0, 1, 2, 3]),
    csv_ddpm_output_path=RESULTS_PATH,
    csv_ddim_output_path=RESULTS_FAST_PATH
)
