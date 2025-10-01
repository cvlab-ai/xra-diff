# TODO: enable once left side dataset is prepared
exit(0)

from reconsnet.data.dataset import ClinicalDataset
from reconsnet.util.test import clinical_test
from reconsnet.model.diffusion import DiffusionModule

CHECKPOINT_PATH = "left.ckpt"
DATA_PATH = "/home/shared/uck-left"
RESULTS_PATH = "data/clinical_diffusion_left.csv"
MODEL = DiffusionModule.load_from_checkpoint(CHECKPOINT_PATH, lr=1e-4)
RECONSTRUCT = lambda x: MODEL.fast_reconstruct(*x, num_inference_steps=10, guidance=True)

clinical_test(
    model=MODEL,
    ds=ClinicalDataset(root_dir=DATA_PATH),
    csv_output_path=RESULTS_PATH,
    reconstruct=RECONSTRUCT
)
