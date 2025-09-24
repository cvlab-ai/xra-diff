from reconsnet.util.test import synthetic_test_adaptive, clinical_test
from reconsnet.model.diffusion import DiffusionModule
from reconsnet.data.dataset import XRayDatasetRight, ClinicalDataset

DATA_PATH = "/home/shared/imagecas/projections_split/val"
CLINICAL_DATA_PATH = "/home/shared/uck-right"
CHECKPOINT_PATH = "stronger-conditioning.ckpt"

print("---------Sampling steps--------------------------")

model = DiffusionModule.load_from_checkpoint(CHECKPOINT_PATH, lr=1e-5)
ds_synthetic = XRayDatasetRight(root_dir=DATA_PATH)
ds_clinical = ClinicalDataset(root_dir=CLINICAL_DATA_PATH)

for steps in [1, 5, 10, 20, 50]:
    reconstruct = lambda x: model.fast_reconstruct(*x, num_inference_steps=steps, guidance=True)
    synthetic_test_adaptive(
        model=model,
        ds=ds_synthetic,
        csv_output_path=f"data/synthetic_sampling_steps_{steps}.csv",
        reconstruct=reconstruct
    )
    clinical_test(
        model=model,
        ds=ds_clinical,
        csv_output_path=f"data/synthetic_sampling_steps_{steps}.csv",
        reconstruct=reconstruct,
    )
