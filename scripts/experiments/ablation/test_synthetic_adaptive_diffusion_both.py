# TODO: enable once both sides training is finished
exit(0)

from reconsnet.util.test import synthetic_test_adaptive
from reconsnet.model.diffusion import DiffusionModule
from reconsnet.data.dataset import XRayDatasetLeft, XRayDatasetRight

CHECKPOINT_PATH = "both.ckpt"
DATA_PATH = "/home/shared/imagecas/projections_split/pilot"
MODEL = DiffusionModule.load_from_checkpoint(CHECKPOINT_PATH, lr=1e-4)
RECONSTRUCT = lambda x: MODEL.fast_reconstruct(*x, num_inference_steps=10, guidance=True)

synthetic_test_adaptive(
    model=MODEL,
    ds=XRayDatasetRight(root_dir=DATA_PATH),
    csv_output_path="data/synthetic_adaptive_diffusion_both_right.csv",
    reconstruct=RECONSTRUCT
)

synthetic_test_adaptive(
    model=MODEL,
    ds=XRayDatasetLeft(root_dir=DATA_PATH),
    csv_output_path="data/synthetic_adaptive_diffusion_both_left.csv",
    reconstruct=RECONSTRUCT
)
