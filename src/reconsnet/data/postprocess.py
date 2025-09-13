import numpy as np
import torch
import torch.nn.functional as F

from scipy.ndimage import convolve


def denoise_pcd(pcd, radius=2, min_neighbors=5):
    diff = pcd[:, np.newaxis, :] - pcd[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    neighbor_count = (dist < radius).sum(axis=1) - 1
    mask = neighbor_count >= min_neighbors
    return pcd[mask]


def denoise_voxels(voxel_grid, min_neighbors=5):
    kernel = torch.ones((1,1,3,3,3), device=voxel_grid.device)
    kernel[0,0,1,1,1] = 0

    neighbor_count = F.conv3d(voxel_grid, kernel, padding=1)
    mask = neighbor_count >= min_neighbors
    voxel_grid_denoised = voxel_grid * mask.float()

    return voxel_grid_denoised
