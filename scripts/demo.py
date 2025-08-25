import numpy as np
import joblib
import matplotlib
import matplotlib.pyplot as plt
import nibabel as nib
import torch

from reconsnet.util.coords import compute_downscales
from reconsnet.data.preprocess import preprocess
from reconsnet.data.dataset import XRayDatasetRight, XRayDatasetLeft

DOWNSAMPLE = 10
GRID_DIM = 128


def grid_to_pointcloud(grid, threshold=0.01):
    coords = torch.argwhere(grid[0] > threshold)
    return coords


def volume_to_pointcloud(path, threshold=0.5):
    nii = nib.load(path)
    volume = nii.get_fdata()
    pcd = np.argwhere(volume > threshold)
    
    step, stepz = compute_downscales(volume.shape, [GRID_DIM, GRID_DIM, GRID_DIM])
    pcd[:, [0, 1]] = pcd[:, [0, 1]] / step
    pcd[:, 2] = pcd[:, 2] / stepz
    return pcd


def visualize_pointclouds(pcs, labels, colors, alphas, s=2):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    for pts, lbl, col, alpha in zip(pcs, labels, colors, alphas):
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                   s=s, c=col, alpha=alpha, label=lbl)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    ax.set_title("Reconstruction vs Original")
    plt.show()

def recons(path):
    projections = joblib.load(path)["projections"]

    xray0 = projections[0]
    xray1 = projections[1]

    grid = preprocess(xray0, xray1, [128, 128, 128])
    points_recon = grid_to_pointcloud(grid, threshold=0.01)[::DOWNSAMPLE]
    return points_recon


matplotlib.use("WebAgg")


ds_right = XRayDatasetRight(root_dir="data/projections")
ds_left = XRayDatasetLeft(root_dir="data/projections")

points_orig = volume_to_pointcloud("data/example.label.nii.gz", threshold=0.5)[::DOWNSAMPLE]

print(ds_right[0][1].shape, ds_left[0][1].shape)

grid_right = grid_to_pointcloud(ds_right[0][0])
grid_left = grid_to_pointcloud(ds_left[0][0])
grid_right_gt = grid_to_pointcloud(ds_right[0][1])
grid_left_gt = grid_to_pointcloud(ds_left[0][1])

visualize_pointclouds(
    [
        grid_right,
        grid_left,
        grid_right_gt,
        grid_left_gt
    ],
    ["Reconstruction (right)", "Reconstruction (left)", "Ground Truth (right)", "Ground Truth (left)"],
    ["blue", "red", "yellow", "green"],
    [1.0, 1.0, 0.3, 0.3],
    s=1
)


