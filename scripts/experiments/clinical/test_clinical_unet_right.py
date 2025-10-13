import torch.nn.functional as F

from reconsnet.data.dataset import ClinicalDataset
from reconsnet.util.test import clinical_test
from reconsnet.model.unet3d import Unet3DModule

CHECKPOINT_PATH = "/home/shared/model-weights/baseline-unet-right.ckpt"
DATA_PATH = "/home/shared/uck-right"
RESULTS_PATH = "data/clinical_unet_right.csv"
MODEL = Unet3DModule.load_from_checkpoint(CHECKPOINT_PATH, lr=1e-5)
RECONSTRUCT = lambda x: F.sigmoid(MODEL.forward(x[0]))

clinical_test(
    model=MODEL,
    ds=ClinicalDataset(root_dir=DATA_PATH),
    csv_output_path=RESULTS_PATH,
    reconstruct=RECONSTRUCT,
    repeat_each=1
)
