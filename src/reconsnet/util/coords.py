import numpy as np
import torch.nn.functional as F

from ..config import get_config


def pcd_to_voxel(pcd, grid_res, downscale=1, downscalez=1):
    voxel_grid = np.zeros(grid_res, dtype=np.uint8)
    pcd[:, [0, 1]] = pcd[:, [0, 1]] / downscale
    pcd[:, 2] = pcd[:, 2] / downscalez
    pcd_vox = np.floor(pcd).astype(int)
    voxel_grid[pcd_vox[:, 0], pcd_vox[:, 1], pcd_vox[:, 2]] = 1
    return voxel_grid


def normalize(pcd, newmax):
    max = pcd.max(axis=0)
    min = pcd.min(axis=0)
    scale = max - min
    pcd = (pcd - min) / scale
    return pcd * (newmax - 1e-7)

def compute_downscales(volume_shape, grid_res):
    return volume_shape[0] / grid_res[0], volume_shape[2] / grid_res[2]

def transpose(img):
    return np.flipud(np.transpose(img, (1, 0)))
    
def pad_pow2(x, target_size=64):
    d, h, w = x.shape[-3:]
    pad_d = target_size - d
    pad_h = target_size - h
    pad_w = target_size - w
    padding = (0, pad_w, 0, pad_h, 0, pad_d)
    return F.pad(x, padding, mode="constant", value=0)

def crop_grid(x):
    grid_dim = get_config()['data']['grid_dim']    
    return x[..., :grid_dim, :grid_dim, :grid_dim]    
