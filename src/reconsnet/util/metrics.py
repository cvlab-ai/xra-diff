import torch
import torch.nn.functional as F

from scipy.spatial import cKDTree

from ..data.postprocess import denoise_voxels


def dice(pred, target, threshold=0.5, eps=1e-8):
    pred_bin = (pred > threshold).float()
    target_bin = (target > 0).float()

    intersection = (pred_bin * target_bin).sum()
    union = pred_bin.sum() + target_bin.sum()
    dice = (2.0 * intersection + eps) / (union + eps)
    return dice


def _downsample(volume, tgt=(60,60,60)):
    volume_down = F.interpolate(volume.float(), size=tgt, mode="trilinear", align_corners=False)
    return volume_down


def confusion(pred, target, threshold, prefix=""):

    
    pred_down = _downsample(pred)
    target_down = _downsample(target)
    
    pred_bin = (pred_down > threshold).float()
    pred_bin = denoise_voxels(pred_bin)
    gt_bin = (target_down > 0).float()
    tp = (pred_bin * gt_bin).sum().item()
    fp = (pred_bin * (1 - gt_bin)).sum().item()
    fn = ((1 - pred_bin) * gt_bin).sum().item()
    tn = ((1 - pred_bin) * (1 - gt_bin)).sum().item()
    dice3d = (2 * tp) / (2 * tp + fp + fn + 1e-8)
    return {
        f"{prefix}dice3d_{threshold}": dice3d,
        f"{prefix}TP_{threshold}": tp, 
        f"{prefix}FP_{threshold}": fp, 
        f"{prefix}FN_{threshold}": fn, 
        f"{prefix}TN_{threshold}": tn
    }


def chamfer_distance(pred, target, threshold, prefix="", to_mm=1.7):
    pred_down = _downsample(pred)
    target_down = _downsample(target)
    
    pred_bin = (pred_down > threshold).float()
    pred_bin = denoise_voxels(pred_bin)
    gt_bin = (target_down > 0).float()
    
    pred_points = pred_bin.nonzero(as_tuple=False).float()
    gt_points = gt_bin.nonzero(as_tuple=False).float()
    
    if pred_points.shape[0] == 0 or gt_points.shape[0] == 0:
        return {f"{prefix}chamfer_{threshold}": 1e7}

    
    dist_matrix = torch.cdist(pred_points, gt_points, p=2)

    min_dist_1_to_2 = torch.min(dist_matrix, dim=1).values * to_mm
    min_dist_2_to_1 = torch.min(dist_matrix, dim=0).values * to_mm
    min_dist_1_to_2 = min_dist_1_to_2**2
    min_dist_2_to_1 = min_dist_2_to_1**2
    dist_1 = torch.mean(min_dist_1_to_2, dim=0)
    dist_2 = torch.mean(min_dist_2_to_1, dim=0)
    chamfer_dist = (dist_1 + dist_2) / 2.0

    return {
        f"{prefix}chamfer_{threshold}": chamfer_dist.item()
    }


def lumen_diameter_error(pred, target):
    pass
