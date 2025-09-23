'''
test on a synthesized dataset
'''
from reconsnet.util.test import synthetic_test_adaptive
from reconsnet.model.diffusion import DiffusionModule
from reconsnet.data.dataset import XRayDatasetRight, default_transform, random_move, random_scale

from torch.utils.data import Subset


CHECKPOINT_PATH = "stronger-conditioning.ckpt"
DATA_PATH = "/home/shared/imagecas/projections_split/val"
RESULTS_PATH = "data/synthetic_motion_right.csv"
MODEL = DiffusionModule.load_from_checkpoint(CHECKPOINT_PATH, lr=1e-4)
RECONSTRUCT = lambda x: MODEL.fast_reconstruct(*x, num_inference_steps=10, guidance=True)

TRANSLATION_RANGE=20 # [px]
SCALING_RANGE=0.2

def motion(projections, gt):
    projections[1].img = random_move(
        random_scale(projections[1].img, SCALING_RANGE), 
        TRANSLATION_RANGE
    )
    
    return default_transform(projections, gt)


synthetic_test_adaptive(
    model=MODEL,
    ds=XRayDatasetRight(root_dir=DATA_PATH, transform=motion),
    csv_output_path=RESULTS_PATH,
    reconstruct=RECONSTRUCT
)
