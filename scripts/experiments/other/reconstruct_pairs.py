import argparse
from pathlib import Path
import torch
from reconsnet.data.postprocess import denoise_voxels, percentile_threshold
from reconsnet.model.diffusion import DiffusionModule
from reconsnet.model.unet3d import Unet3DModule
from reconsnet.data.dataset import load_clinical_sample, default_transform
from reconsnet.util.coords import reproject
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


PAIRS_DIR=Path("/home/shared/xradiffpdataset")
UNET_PATH = "/home/shared/model-weights/baseline-unet-both.ckpt"
DIFFUSION_PATH = "/home/shared/model-weights/both.ckpt"
MODEL_OUTDIR = Path("/home/shared/xradiffpdatasetmodeloutput")

matplotlib.use("WebAgg")
matplotlib.rcParams["webagg.port"] = 2137
matplotlib.rcParams['webagg.open_in_browser'] = False


def run_reconstruction(model, reconstruct, xray0, xray1):
    backproj, _, img0, img1 = default_transform([xray0, xray1], None)

    backproj = backproj.to(model.device).unsqueeze(0)
    img0 = img0.to(model.device).unsqueeze(0).unsqueeze(0)
    img1 = img1.to(model.device).unsqueeze(0).unsqueeze(0)

    sample = (backproj, img0, img1)
    pred_vox = reconstruct(sample)
    threshold = percentile_threshold(pred_vox, percentile=0.99)
    hat_bin = denoise_voxels((pred_vox > threshold).float(), min_neighbors=5).squeeze().cpu().numpy()
    pcd = np.argwhere(hat_bin > 0)
    return pcd


def main():
    parser = argparse.ArgumentParser()

    MODEL_OUTDIR.mkdir(parents=True, exist_ok=True)

    diffusion_model = DiffusionModule.load_from_checkpoint(DIFFUSION_PATH, lr=1e-5)
    diffusion_model.eval()
    difffusion_reconstruct = lambda x: diffusion_model.fast_reconstruct(*x, num_inference_steps=10, guidance=True)

    # diffusion_reconstruct = 
    unet_model = Unet3DModule.load_from_checkpoint(UNET_PATH, lr=1e-5)
    unet_model.eval()

    for pair_dir in sorted(PAIRS_DIR.iterdir()):
        if not pair_dir.is_dir():
            continue

        xray0_path = pair_dir / "0.binary.png"
        xray1_path = pair_dir / "1.binary.png"
        xray0_params = pair_dir / "0.metadata.json"
        xray1_params = pair_dir / "1.metadata.json"

        xray0 = load_clinical_sample(xray0_params, xray0_path, pairs=True)
        xray1 = load_clinical_sample(xray1_params, xray1_path, pairs=True)

        output_pair_dir = PAIRS_DIR / pair_dir.name
        output_pair_dir.mkdir(exist_ok=True)
        
        diffusion_pcd = run_reconstruction(diffusion_model, difffusion_reconstruct, xray0, xray1)

        fig = plt.figure(figsize=(15,15))
        ax = fig.add_subplot(111, projection='3d')
        plt.subplots_adjust(bottom=0.2)
        ax.scatter(diffusion_pcd[:,0], diffusion_pcd[:,1], diffusion_pcd[:,2],
                            c='black', s=1, alpha=0.7, label="DIFFUSION")

        ax.set_axis_off()
        ax.legend(loc='upper right')
        plt.show()
        # unet_pcd = run
        
        # diffusion_pred = run_reconstruction(diffusion_model, xray0, xray1)
        # unet_pred = run_reconstruction(unet_model, xray0, xray1)


        # np.save(output_pair_dir / "diffusion_0.npy", diff0)
        # np.save(output_pair_dir / "diffusion_1.npy", diff1)
        # np.save(output_pair_dir / "unet_0.npy", unet0)
        # np.save(output_pair_dir / "unet_1.npy", unet1)




if __name__ == "__main__":
    main()
