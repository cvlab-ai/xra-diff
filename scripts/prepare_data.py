import nibabel as nib
import numpy as np
import sys
import yaml
import matplotlib.pyplot as plt
import torch
import joblib

from pathlib import Path
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from random import uniform
from reconsnet.util.camera import build_camera_model
from reconsnet.util.coords import pcd_to_voxel, compute_downscales, transpose
from reconsnet.data.data import XRay


AVOID_UNNECESSARY_COPY = True

INPUT_PATH=Path("/home/shared/imagecas/imagecas_unzipped")
# INPUT_PATH = Path("data/")
OUTPUT_PATH=Path("data/projections")
CONFIG_PATH=Path("config")
with open(CONFIG_PATH / "projections.yaml") as f:
    CONFIG = yaml.SafeLoader(f.read()).get_data()

def main():
    OUTPUT_PATH.mkdir(exist_ok=True)
    files = list(INPUT_PATH.glob('**/*.label.nii.gz'))
    
    for i, path in enumerate(tqdm(files)):
        grid_left, grid_right = load_data(path)

        joblib.dump(
            { 
                "gt": grid_right, 
                "projections": model_projections(CONFIG['right'], grid_right)
            },
            OUTPUT_PATH / f"right{i}.joblib"
        )
        joblib.dump(
            {
                "gt": grid_left,
                "projections": model_projections(CONFIG['left'], grid_left),
            },
            OUTPUT_PATH / f"left{i}.joblib"
        )


def model_projections(side_config: dict, grid: np.array):
    global_config = CONFIG['global']
    img_res = global_config['image_resolution']
    grid_res = global_config['grid_resolution']

    projections = []
    for proj_ix in side_config.keys():
        local_config = side_config[proj_ix]
  
        sid = uniform(*local_config['sid'])
        sod = uniform(*local_config['sod'])
        alpha = np.deg2rad(uniform(*local_config['alpha']))
        beta = np.deg2rad(uniform(*local_config['beta']))
        grid_spacing = uniform(*global_config["grid_spacing"]) 
        img_spacing = uniform(*global_config["image_spacing"])
        camera = build_camera_model(alpha, beta, sid, sod, grid_spacing, grid_res, img_spacing, img_res)
        projection = camera(grid)[0]
        
        projections.append(
            XRay(
                projection, 
                sid, sod,
                alpha, beta,
                img_spacing, img_res
            )
        )
        
    return projections


def load_data(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    grid_res = CONFIG['global']['grid_resolution']

    volume = nib.load(path)
    volume = volume.get_fdata()
    pcd = np.argwhere(volume > 0)
    pcd_left, pcd_right = split_left_right(pcd)
    step, stepz = compute_downscales(volume.shape, grid_res)
   
    return pcd_to_voxel(pcd_left, grid_res, step, stepz), pcd_to_voxel(pcd_right, grid_res, step, stepz)


def split_left_right(coords):
    if len(coords) == 0: return coords, coords
    clus = DBSCAN(eps=5, min_samples=5)
    clus.fit(coords)
    labels = clus.labels_
    if len(labels) == 1: return coords, coords
    coords0 = coords[labels == 0]
    coords1 = coords[labels == 1]
    centroid0 = coords0.mean(axis=0)
    centroid1 = coords1.mean(axis=0)
    if centroid0[1] > centroid1[1]: coords0, coords1
    return coords1, coords0


def example():
    import matplotlib
    matplotlib.use("WebAgg")
    grid_left, grid_right = load_data("data/example.label.nii.gz")
    
    for _ in range(0, 5):
    
        right_projections = model_projections(CONFIG["right"], grid_right)
        left_projections = model_projections(CONFIG["left"], grid_left)

        _, axes = plt.subplots(1, 5, figsize=(15, 5))
        vol_proj = np.max(grid_left + grid_right, axis=0) 
        
        axes[0].imshow(vol_proj.T, cmap="gray", origin="lower")
        axes[0].set_title("3D Volume max. intensity projection")
        axes[1].imshow(transpose(right_projections[0].img), cmap="gray")
        axes[1].set_title("right proj 0")
        axes[2].imshow(transpose(left_projections[0].img), cmap="gray")
        axes[2].set_title("left proj 0")
        axes[3].imshow(transpose(right_projections[1].img), cmap="gray")
        axes[3].set_title("right proj 1")
        axes[4].imshow(transpose(left_projections[1].img), cmap="gray")
        axes[4].set_title("left proj 1")

        for ax in axes:
            ax.axis("off")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":    
    if '--example' in sys.argv:
        example()
    else:
        main()
