import numpy as np


def denoise_pcd(pcd, radius=2, min_neighbors=5):
    diff = pcd[:, np.newaxis, :] - pcd[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    neighbor_count = (dist < radius).sum(axis=1) - 1
    mask = neighbor_count >= min_neighbors
    return pcd[mask]
