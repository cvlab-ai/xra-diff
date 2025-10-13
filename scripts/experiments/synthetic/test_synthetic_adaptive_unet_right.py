import torch.nn.functional as F

from reconsnet.util.test import synthetic_test_adaptive
from reconsnet.model.unet3d import Unet3DModule
from reconsnet.data.dataset import XRayDatasetRight

CHECKPOINT_PATH = "/home/shared/model-weights/baseline-unet-right.ckpt"
DATA_PATH = "/home/shared/imagecas/projections_split/pilot"
RESULTS_PATH = "data/synthetic_adaptive_unet_right.csv"
MODEL = Unet3DModule.load_from_checkpoint(CHECKPOINT_PATH, lr=1e-5)
RECONSTRUCT = lambda x: F.sigmoid(MODEL.forward(x[0]))

synthetic_test_adaptive(
    model=MODEL,
    ds=XRayDatasetRight(root_dir=DATA_PATH),
    csv_output_path=RESULTS_PATH,
    reconstruct=RECONSTRUCT,
    repeat_each=1
)
