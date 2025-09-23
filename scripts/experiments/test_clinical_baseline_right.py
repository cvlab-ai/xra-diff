import torch.nn.functional as F

from reconsnet.util.test import clinical_test
from reconsnet.model.gan import GANModule
from reconsnet.data.dataset import ClinicalDataset

from torch.utils.data import Subset

CHECKPOINT_PATH = "baseline.ckpt"
DATA_PATH = "/home/shared/uck-right"
RESULTS_PATH = "data/clinical_baseline_right.csv"
MODEL = GANModule.load_from_checkpoint(CHECKPOINT_PATH)
RECONSTRUCT = lambda x: MODEL.generator.forward(x[0])

clinical_test(
    model=MODEL,
    ds=ClinicalDataset(root_dir=DATA_PATH),
    csv_output_path=RESULTS_PATH,
    reconstruct=RECONSTRUCT
)

