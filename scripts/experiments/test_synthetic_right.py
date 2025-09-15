'''
test on a synthesized dataset
'''
from reconsnet.util.test import synthetic_test
from reconsnet.model.diffusion import DiffusionModule
from reconsnet.data.dataset import XRayDatasetRight

from torch.utils.data import Subset


CHECKPOINT_PATH = "better-backproj.ckpt"
DATA_PATH = "/home/shared/imagecas/projections"
RESULTS_PATH = "data/synthetic_right.csv"
MODEL = DiffusionModule.load_from_checkpoint(CHECKPOINT_PATH, lr=1e-4)
RECONSTRUCT = lambda x: MODEL.fast_reconstruct(x, num_inference_steps=10, guidance=True)

synthetic_test(
    model=MODEL,
    ds=Subset(XRayDatasetRight(root_dir=DATA_PATH), [0, 1, 2, 3]),
    csv_output_path=RESULTS_PATH,
    reconstruct=RECONSTRUCT
)
