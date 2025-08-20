import numpy as np
import joblib
import matplotlib
import matplotlib.pyplot as plt
import nibabel as nib

from reconsnet.data.preprocess import preprocess
from reconsnet.data.dataset import XRay

DOWNSAMPLE = 100


def grid_to_pointcloud(grid, threshold=0.01):
    coords = np.argwhere(grid > threshold)
    return coords


def volume_to_pointcloud(path, threshold=0.5):
    nii = nib.load(path)
    data = nii.get_fdata()
    coords = np.argwhere(data > threshold)

    coords_hom = np.c_[coords, np.ones(len(coords))]
    coords_world = coords_hom @ nii.affine.T
    return coords_world[:, :3]


def visualize_pointclouds(pcs, labels, colors, s=2):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    for pts, lbl, col in zip(pcs, labels, colors):
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                   s=s, c=col, alpha=0.5, label=lbl)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    ax.set_title("Reconstruction vs Original")
    plt.show()


matplotlib.use("WebAgg")

projections = joblib.load("data/projections/right0.joblib")

xray0 = projections[0]
xray1 = projections[1]

grid = preprocess(xray0, xray1, [128, 128, 128])
points_recon = grid_to_pointcloud(grid, threshold=0.01)[::DOWNSAMPLE]
points_orig = volume_to_pointcloud("data/example.label.nii.gz", threshold=0.5)[::DOWNSAMPLE]

visualize_pointclouds(
    [points_recon, points_orig],
    ["Reconstruction", "Ground Truth"],
    ["red", "blue"],
    s=1
)


