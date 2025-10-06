from reconsnet.util.test import synthetic_test_adaptive, clinical_test
from reconsnet.model.diffusion import DiffusionModule
from reconsnet.data.dataset import XRayDatasetLeft, XRayDatasetRight, ClinicalDataset

CHECKPOINT_PATH = "/home/shared/model-weights/both.ckpt"
DATA_PATH = "/home/shared/imagecas/projections_split/pilot"
CLINICAL_DATA_RIGHT_PATH = "/home/shared/uck-right"
CLINICAL_DATA_LEFT_PATH = "/home/shared/uck-left"
MODEL = DiffusionModule.load_from_checkpoint(CHECKPOINT_PATH, lr=1e-4)
RECONSTRUCT = lambda x: MODEL.fast_reconstruct(*x, num_inference_steps=10, guidance=True)


synthetic_test_adaptive(
    model=MODEL,
    ds=XRayDatasetRight(root_dir=DATA_PATH),
    csv_output_path="data/synthetic_both_right.csv",
    reconstruct=RECONSTRUCT
)

synthetic_test_adaptive(
    model=MODEL,
    ds=XRayDatasetLeft(root_dir=DATA_PATH),
    csv_output_path="data/synthetic_both_left.csv",
    reconstruct=RECONSTRUCT
)

# #NOTE: disabled for pilot study
# clinical_test(
#     model=MODEL,
#     ds=ClinicalDataset(root_dir=CLINICAL_DATA_RIGHT_PATH),
#     csv_output_path=f"data/clinical_both_right.csv",
#     reconstruct=RECONSTRUCT,
# )

# #NOTE: disabled for pilot study
# clinical_test(
#     model=MODEL,
#     ds=ClinicalDataset(root_dir=CLINICAL_DATA_LEFT_PATH),
#     csv_output_path=f"data/clinical_both_left.csv",
#     reconstruct=RECONSTRUCT,
# )
