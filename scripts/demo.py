import numpy as np
import joblib
import matplotlib
import matplotlib.pyplot as plt
import nibabel as nib

from reconsnet.util.coords import pcd_to_voxel, compute_downscales
from reconsnet.data.preprocess import preprocess
from reconsnet.data.dataset import get_dm_left, get_dm_right

DOWNSAMPLE = 100
GRID_DIM = 128


def grid_to_pointcloud(grid, threshold=0.01):
    coords = np.argwhere(grid > threshold)
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
    projections = joblib.load(path)

    xray0 = projections[0]
    xray1 = projections[1]

    grid = preprocess(xray0, xray1, [128, 128, 128])
    points_recon = grid_to_pointcloud(grid, threshold=0.01)[::DOWNSAMPLE]
    return points_recon


matplotlib.use("WebAgg")

points_recons_left = recons("data/projections/left0.joblib")
points_recons_right = recons("data/projections/right0.joblib")
points_orig = volume_to_pointcloud("data/example.label.nii.gz", threshold=0.5)[::DOWNSAMPLE]

visualize_pointclouds(
    [points_recons_right, points_recons_left, points_orig],
    ["Reconstruction (right)", "Reconstruction (left)", "Ground Truth"],
    ["blue", "red", "yellow"],
    [1.0, 1.0, 0.3],
    s=1
)

dm0, dm1 = get_dm_right("data/"), get_dm_left("data/")


