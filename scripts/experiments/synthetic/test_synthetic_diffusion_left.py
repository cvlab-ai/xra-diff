'''
test on a synthesized dataset
'''
from reconsnet.util.test import synthetic_test
from reconsnet.model.diffusion import DiffusionModule
from reconsnet.data.dataset import XRayDatasetLeft

CHECKPOINT_PATH = "left.ckpt"
DATA_PATH = "/home/shared/imagecas/projections_split/pilot"
RESULTS_PATH = "data/synthetic_diffusion_left.csv"
MODEL = DiffusionModule.load_from_checkpoint(CHECKPOINT_PATH, lr=1e-4)
RECONSTRUCT = lambda x: MODEL.fast_reconstruct(*x, num_inference_steps=10, guidance=True)

synthetic_test(
    model=MODEL,
    ds=XRayDatasetLeft(root_dir=DATA_PATH),
    csv_output_path=RESULTS_PATH,
    reconstruct=RECONSTRUCT
)
