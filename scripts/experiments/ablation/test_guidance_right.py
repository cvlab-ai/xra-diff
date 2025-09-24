from reconsnet.util.test import synthetic_test_adaptive, clinical_test
from reconsnet.model.diffusion import DiffusionModule
from reconsnet.data.dataset import XRayDatasetRight, ClinicalDataset

DATA_PATH = "/home/shared/imagecas/projections_split/val"
CLINICAL_DATA_PATH = "/home/shared/uck-right"
CHECKPOINT_PATH = "stronger-conditioning.ckpt"

print("---------Guidance scale--------------------------")

model = DiffusionModule.load_from_checkpoint(CHECKPOINT_PATH, lr=1e-5)
ds_synthetic = XRayDatasetRight(root_dir=DATA_PATH)
ds_clinical = ClinicalDataset(root_dir=CLINICAL_DATA_PATH)
reconstruct = lambda x: model.fast_reconstruct(*x, num_inference_steps=10, guidance=True)

for gsc in [0.0, 1.0, 2.0, 6.0, 8.0, 10.0, 12.0]:
    model.guidance_scale = gsc
    synthetic_test_adaptive(
        model=model,
        ds=ds_synthetic,
        csv_output_path=f"data/synthetic_guidance_scale_{gsc}.csv",
        reconstruct=reconstruct
    )
    clinical_test(
        model=model,
        ds=ds_clinical,
        csv_output_path=f"data/clinical_guidance_scale_{gsc}.csv",
        reconstruct=reconstruct,
    )
