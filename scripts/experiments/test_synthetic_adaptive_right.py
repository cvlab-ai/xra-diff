'''
test on a synthesized dataset
'''
from reconsnet.util.test import synthetic_test_adaptive
from reconsnet.model.diffusion import DiffusionModule
from reconsnet.data.dataset import XRayDatasetRight

CHECKPOINT_PATH = "stronger-conditioning.ckpt"
DATA_PATH = "/home/shared/imagecas/projections_split/val"
RESULTS_PATH = "data/synthetic_adaptive_right.csv"
MODEL = DiffusionModule.load_from_checkpoint(CHECKPOINT_PATH, lr=1e-4)
RECONSTRUCT = lambda x: MODEL.fast_reconstruct(*x, num_inference_steps=10, guidance=True)

synthetic_test_adaptive(
    model=MODEL,
    ds=XRayDatasetRight(root_dir=DATA_PATH),
    csv_output_path=RESULTS_PATH,
    reconstruct=RECONSTRUCT
)
