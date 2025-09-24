'''
test on a synthesized dataset
'''
from reconsnet.util.test import synthetic_test_adaptive
from reconsnet.model.gan import GANModule
from reconsnet.data.dataset import XRayDatasetRight

CHECKPOINT_PATH = "baseline.ckpt"
DATA_PATH = "/home/shared/imagecas/projections_split/val"
RESULTS_PATH = "data/synthetic_adaptive_baseline_right.csv"
MODEL = GANModule.load_from_checkpoint(CHECKPOINT_PATH)
RECONSTRUCT = lambda x: MODEL.generator.forward(x[0])


synthetic_test_adaptive(
    model=MODEL,
    ds=XRayDatasetRight(root_dir=DATA_PATH),
    csv_output_path=RESULTS_PATH,
    reconstruct=RECONSTRUCT
)
