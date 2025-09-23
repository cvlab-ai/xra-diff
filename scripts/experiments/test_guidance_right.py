from reconsnet.util.test import synthetic_test_adaptive
from reconsnet.model.diffusion import DiffusionModule
from reconsnet.data.dataset import XRayDatasetRight

DATA_PATH = "/home/shared/imagecas/projections_split/val"
CHECKPOINT_PATH = "stronger-conditioning.ckpt"

print("---------Guidance scale--------------------------")

model = DiffusionModule.load_from_checkpoint(CHECKPOINT_PATH, lr=1e-5)
ds = XRayDatasetRight(root_dir=DATA_PATH)
reconstruct = lambda x: model.fast_reconstruct(*x, num_inference_steps=10, guidance=True)

for gsc in [0.0, 1.0, 2.0, 6.0, 8.0, 10.0, 12.0]:
    model.guidance_scale = gsc
    synthetic_test_adaptive(
        model=model,
        ds=ds,
        csv_output_path=f"data/guidance_scale_{gsc}.csv",
        reconstruct=reconstruct
    )
