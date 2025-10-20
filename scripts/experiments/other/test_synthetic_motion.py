from reconsnet.util.test import synthetic_test_adaptive
from reconsnet.model.diffusion import DiffusionModule
from reconsnet.data.dataset import XRayDatasetRight, XRayDatasetLeft, default_transform, random_move, random_scale

DATA_PATH = "/home/shared/imagecas/projections_split/val"
CHECKPOINT_RIGHT_PATH = "/home/shared/model-weights/right.ckpt"
CHECKPOINT_LEFT_PATH = "/home/shared/model-weights/left.ckpt"
MODEL_RIGHT = DiffusionModule.load_from_checkpoint(CHECKPOINT_RIGHT_PATH, lr=1e-4)
MODEL_LEFT = DiffusionModule.load_from_checkpoint(CHECKPOINT_LEFT_PATH, lr=1e-4)
RECONSTRUCT_RIGHT = lambda x: MODEL_RIGHT.fast_reconstruct(*x, num_inference_steps=10, guidance=True)
RECONSTRUCT_LEFT = lambda x: MODEL_LEFT.fast_reconstruct(*x, num_inference_steps=10, guidance=True)
TRANSLATION_RANGE=20 # [px]
SCALING_RANGE=0.2

def motion(projections, gt):
    projections[1].img = random_move(
        random_scale(projections[1].img, SCALING_RANGE), 
        TRANSLATION_RANGE
    )
    
    return default_transform(projections, gt)


synthetic_test_adaptive(
    model=MODEL_RIGHT,
    ds=XRayDatasetRight(root_dir=DATA_PATH, transform=motion),
    csv_output_path="data/synthetic_motion_right.csv",
    reconstruct=RECONSTRUCT_RIGHT
)

synthetic_test_adaptive(
    model=MODEL_LEFT,
    ds=XRayDatasetLeft(root_dir=DATA_PATH, transform=motion),
    csv_output_path="data/synthetic_motion_left.csv",
    reconstruct=RECONSTRUCT_LEFT
)

