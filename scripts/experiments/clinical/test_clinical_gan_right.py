from reconsnet.util.test import clinical_test
from reconsnet.model.gan import GANModule
from reconsnet.data.dataset import ClinicalDataset

CHECKPOINT_PATH = "/home/shared/model-weights/baseline-gan-right.ckpt"
DATA_PATH = "/home/shared/uck-right"
RESULTS_PATH = "data/clinical_gan_right.csv"
MODEL = GANModule.load_from_checkpoint(CHECKPOINT_PATH)
RECONSTRUCT = lambda x: MODEL.generator.forward(x[0])

clinical_test(
    model=MODEL,
    ds=ClinicalDataset(root_dir=DATA_PATH),
    csv_output_path=RESULTS_PATH,
    reconstruct=RECONSTRUCT
)

