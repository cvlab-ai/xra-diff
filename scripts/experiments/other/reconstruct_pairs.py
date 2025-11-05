import torch.nn.functional as F
import numpy as np

from pathlib import Path
from tqdm import tqdm
from reconsnet.data.postprocess import denoise_voxels, percentile_threshold
from reconsnet.model.diffusion import DiffusionModule
from reconsnet.model.unet3d import Unet3DModule
from reconsnet.data.dataset import load_clinical_sample, default_transform
from reconsnet.util.coords import reproject


PAIRS_DIR=Path("/home/shared/xradiffpdataset")
UNET_PATH = "/home/shared/model-weights/baseline-unet-both.ckpt"
DIFFUSION_PATH = "/home/shared/model-weights/both.ckpt"
MODEL_OUTDIR = Path("/home/shared/xradiffpdatasetmodeloutput")



def run_reconstruction(model, reconstruct, xray0, xray1):
    backproj, _, img0, img1 = default_transform([xray0, xray1], None)

    backproj = backproj.to(model.device).unsqueeze(0)
    img0 = img0.to(model.device).unsqueeze(0).unsqueeze(0)
    img1 = img1.to(model.device).unsqueeze(0).unsqueeze(0)

    sample = (backproj, img0, img1)
    pred_vox = reconstruct(sample)
    threshold = percentile_threshold(pred_vox, percentile=0.995)
    hat_bin = denoise_voxels((pred_vox > threshold).float(), min_neighbors=5).squeeze().cpu().numpy()
    pcd = np.argwhere(hat_bin > 0)
    return pcd


def main():
    MODEL_OUTDIR.mkdir(parents=True, exist_ok=True)
    diffusion_outdir = MODEL_OUTDIR / "diffusion"
    unet_outdir = MODEL_OUTDIR / "unet"
    
    diffusion_outdir.mkdir(exist_ok=True)
    unet_outdir.mkdir(exist_ok=True)

    diffusion_model = DiffusionModule.load_from_checkpoint(DIFFUSION_PATH, lr=1e-5)
    diffusion_model.eval()
    diffusion_reconstruct = lambda x: diffusion_model.fast_reconstruct(*x, num_inference_steps=10, guidance=True)

    unet_model = Unet3DModule.load_from_checkpoint(UNET_PATH, lr=1e-5)
    unet_model.eval()
    unet_reconstruct = lambda x: F.sigmoid(unet_model.forward(x[0]))

    for pair_dir in tqdm(sorted(PAIRS_DIR.iterdir())):
        if not pair_dir.is_dir():
            continue

        xray0_path = pair_dir / "0.binary.png"
        xray1_path = pair_dir / "1.binary.png"
        xray0_params = pair_dir / "0.metadata.json"
        xray1_params = pair_dir / "1.metadata.json"
        diffusion_outpath = diffusion_outdir / f"{pair_dir.name}"
        unet_outpath = unet_outdir / f"{pair_dir.name}"

        xray0 = load_clinical_sample(xray0_params, xray0_path, pairs=True)
        xray1 = load_clinical_sample(xray1_params, xray1_path, pairs=True)

        diffusion_pcd = run_reconstruction(diffusion_model, diffusion_reconstruct, xray0, xray1)
        unet_pcd = run_reconstruction(unet_model, unet_reconstruct, xray0, xray1)

        np.save(diffusion_outpath, diffusion_pcd)
        np.save(unet_outpath, unet_pcd)



if __name__ == "__main__":
    main()
