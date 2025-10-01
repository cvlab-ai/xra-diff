from reconsnet.util.test import synthetic_test_adaptive, clinical_test
from reconsnet.model.diffusion import DiffusionModule
from reconsnet.data.dataset import XRayDatasetRight, ClinicalDataset

DATA_PATH = "/home/shared/imagecas/projections_split/pilot"
CLINICAL_DATA_RIGHT_PATH = "/home/shared/uck-right"
CLINICAL_DATA_LEFT_PATH = "/home/shared/uck-left"
CHECKPOINT_RIGHT_PATH = "stronger-conditioning.ckpt"
CHECKPOINT_LEFT_PATH = "left.ckpt"

print("---------Guidance scale--------------------------")

model_right = DiffusionModule.load_from_checkpoint(CHECKPOINT_RIGHT_PATH, lr=1e-5)
model_left = DiffusionModule.load_from_checkpoint(CHECKPOINT_LEFT_PATH, lr=1e-5)
ds_synthetic_right = XRayDatasetRight(root_dir=DATA_PATH)
ds_clinical_right = ClinicalDataset(root_dir=CLINICAL_DATA_RIGHT_PATH)
ds_synthetic_left = XRayDatasetRight(root_dir=DATA_PATH)
# ds_clinical_left = ClinicalDataset(root_dir=CLINICAL_DATA_LEFT_PATH)
reconstruct_right = lambda x: model_right.fast_reconstruct(*x, num_inference_steps=10, guidance=True)
reconstruct_left = lambda x: model_left.fast_reconstruct(*x, num_inference_steps=10, guidance=True)

for gsc in [0.0, 1.0, 2.0, 6.0, 8.0, 10.0, 12.0]:
    model_right.guidance_scale = gsc
    model_left.guidance_scale = gsc
    synthetic_test_adaptive(
        model=model_right,
        ds=ds_synthetic_right,
        csv_output_path=f"data/synthetic_guidance_scale_{gsc}_right.csv",
        reconstruct=reconstruct_right
    )
    # NOTE: disabled for pilot study
    # clinical_test(
    #     model=model_right,
    #     ds=ds_clinical_right,
    #     csv_output_path=f"data/clinical_guidance_scale_{gsc}_right.csv",
    #     reconstruct=reconstruct_right,
    # )
    synthetic_test_adaptive(
        model=model_left,
        ds=ds_synthetic_left,
        csv_output_path=f"data/synthetic_guidance_scale_{gsc}_left.csv",
        reconstruct=reconstruct_left
    )
    # TODO: enable once left side dataset is prepared
    # clinical_test(
    #     model=model_left,
    #     ds=ds_clinical_left,
    #     csv_output_path=f"data/clinical_guidance_scale_{gsc}_left.csv",
    #     reconstruct=reconstruct_left,
    # )
