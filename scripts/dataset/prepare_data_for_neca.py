import nibabel as nib
import numpy as np
import sys
import yaml
import matplotlib.pyplot as plt
import torch
import joblib
import os
import yaml
import argparse

from pathlib import Path
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from random import uniform
from reconsnet.util.camera import build_camera_model
from reconsnet.util.coords import pcd_to_voxel, compute_downscales, transpose
from reconsnet.data.data import XRay


AVOID_UNNECESSARY_COPY = True

INPUT_PATH = Path("/home/shared/imagecas/imagecas_unzipped")
# INPUT_PATH = Path("data/")
OUTPUT_PATH = Path("data/neca")
CONFIG_PATH = Path("config")
with open(CONFIG_PATH / "projections.yaml") as f:
    CONFIG = yaml.SafeLoader(f.read()).get_data()


def main(max_files=None):
    OUTPUT_PATH.mkdir(exist_ok=True)
    files = list(INPUT_PATH.glob("**/*.label.nii.gz"))

    if max_files is not None:
        files = files[:max_files]
        print(f"Processing {len(files)} files (limited to first {max_files})")
    else:
        print(f"Processing all {len(files)} files")

    for i, path in enumerate(tqdm(files)):
        grids = load_data(path)

        for idx, grid in enumerate(grids):
            projections, settings = model_projections(
                CONFIG["right"] if idx == 1 else CONFIG["left"], grid
            )

            projection_imgs = [proj.img for proj in projections]
            projections_array = np.stack(projection_imgs, axis=0)
            projections_array = np.expand_dims(projections_array, axis=0)

            PROJECTION_OUTPUT_DIR = OUTPUT_PATH / f"{i}_{idx}"

            os.makedirs(PROJECTION_OUTPUT_DIR, exist_ok=True)

            PROJECTION_OUTPUT_PATH = PROJECTION_OUTPUT_DIR / "projections.npy"

            np.save(PROJECTION_OUTPUT_PATH, projections_array)

            for proj_idx, proj_img in enumerate(projection_imgs):
                if hasattr(proj_img, "array"):
                    proj_array = proj_img.array
                elif hasattr(proj_img, "data"):
                    proj_array = proj_img.data
                else:
                    proj_array = np.asarray(proj_img)

                proj_normalized = (proj_array - proj_array.min()) / (
                    proj_array.max() - proj_array.min()
                )

                img_output_path = PROJECTION_OUTPUT_DIR / f"projection_{proj_idx}.png"
                plt.figure(figsize=(8, 8))
                plt.imshow(proj_normalized, cmap="gray")
                plt.axis("off")
                plt.tight_layout()
                plt.savefig(img_output_path, bbox_inches="tight", pad_inches=0, dpi=150)
                plt.close()

            settings_output = {
                "datadir": str(PROJECTION_OUTPUT_PATH),
                "numTrain": 2,
                "global": {
                    "image_resolution": CONFIG["global"]["image_resolution"],
                    "grid_resolution": CONFIG["global"]["grid_resolution"],
                    "image_spacing": float(
                        np.mean(
                            [settings[0]["img_spacing"], settings[1]["img_spacing"]]
                        )
                    ),
                    "grid_spacing": float(settings[0]["grid_spacing"]),
                },
                "projections_settings": [
                    {
                        "alpha": float(np.rad2deg(settings[0]["alpha"])),
                        "beta": float(np.rad2deg(settings[0]["beta"])),
                        "sid": float(settings[0]["sid"]),
                        "sod": float(settings[0]["sod"]),
                    },
                    {
                        "alpha": float(np.rad2deg(settings[1]["alpha"])),
                        "beta": float(np.rad2deg(settings[1]["beta"])),
                        "sid": float(settings[1]["sid"]),
                        "sod": float(settings[1]["sod"]),
                    },
                ],
            }
            PROJECTION_OUTPUT_SETTINGS_PATH = PROJECTION_OUTPUT_DIR / "settings.yaml"
            yaml.dump(settings_output, open(PROJECTION_OUTPUT_SETTINGS_PATH, "w+"))

            PROJECTION_GRID_OUTPUT_PATH = PROJECTION_OUTPUT_DIR / "gt.npy"
            np.save(PROJECTION_GRID_OUTPUT_PATH, grid)

            PROJECTION_CONFIG_PATH = PROJECTION_OUTPUT_DIR / "config.yaml"
            config = yaml.safe_load(open(CONFIG_PATH / "CCTA_base.yaml"))
            config["exp"]["dataconfig"] = str(PROJECTION_OUTPUT_SETTINGS_PATH)
            LOGS_PATH = PROJECTION_OUTPUT_DIR / "logs"
            config["exp"]["expdir"] = str(LOGS_PATH)
            config["exp"]["output_path"] = str(PROJECTION_OUTPUT_DIR / "pred.npy")

            yaml.dump(config, open(PROJECTION_CONFIG_PATH, "w+"))


def model_projections(side_config: dict, grid: np.ndarray):
    global_config = CONFIG["global"]
    img_res = global_config["image_resolution"]
    grid_resolution = global_config["grid_resolution"]

    projections = []
    settings = []
    for proj_ix in side_config.keys():
        local_config = side_config[proj_ix]

        sid = uniform(*local_config["sid"])
        sod = uniform(*local_config["sod"])
        alpha = np.deg2rad(uniform(*local_config["alpha"]))
        beta = np.deg2rad(uniform(*local_config["beta"]))
        grid_spacing = global_config["grid_spacing"][0]
        img_spacing = uniform(*global_config["image_spacing"])
        camera = build_camera_model(
            alpha, beta, sid, sod, grid_spacing, grid_resolution, img_spacing, img_res
        )
        projection = camera(grid)[0]

        settings.append(
            {
                "sid": sid,
                "sod": sod,
                "alpha": alpha,
                "beta": beta,
                "grid_spacing": grid_spacing,
                "img_spacing": img_spacing,
            }
        )

        projections.append(
            XRay(projection, sid, sod, alpha, beta, img_spacing, img_res)
        )

    return projections, settings


def load_data(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    grid_res = CONFIG["global"]["grid_resolution"]

    volume = nib.load(path)
    volume = volume.get_fdata()
    pcd = np.argwhere(volume > 0)
    pcd_left, pcd_right = split_left_right(pcd)
    step, stepz = compute_downscales(volume.shape, grid_res)

    return pcd_to_voxel(pcd_left, grid_res, step, stepz), pcd_to_voxel(
        pcd_right, grid_res, step, stepz
    )


def split_left_right(coords):
    if len(coords) == 0:
        return coords, coords
    clus = DBSCAN(eps=5, min_samples=5)
    clus.fit(coords)
    labels = clus.labels_
    if len(labels) == 1:
        return coords, coords
    coords0 = coords[labels == 0]
    coords1 = coords[labels == 1]
    centroid0 = coords0.mean(axis=0)
    centroid1 = coords1.mean(axis=0)
    if centroid0[1] > centroid1[1]:
        coords0, coords1
    return coords1, coords0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data for NeCA training")
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum number of files to process (default: process all files)",
    )
    args = parser.parse_args()

    main(max_files=args.max_files)
