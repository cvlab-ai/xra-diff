import torch
import torch.nn.functional as F

from scipy.spatial import cKDTree

from ..data.postprocess import denoise_voxels


def dice(pred, target, threshold=0.5, eps=1e-8):
    pred_bin = (pred > threshold).float()
    target_bin = (target > 0).float()

    print("SUMS", pred_bin.sum(), target_bin.sum())

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


def chamfer_distance(pred, target, threshold, prefix=""):
    pred_down = _downsample(pred)
    target_down = _downsample(target)
    
    pred_bin = (pred_down > threshold).float()
    pred_bin = denoise_voxels(pred_bin)
    gt_bin = (target_down > 0).float()
    
    pred_points = pred_bin.nonzero(as_tuple=False)
    gt_points = gt_bin.nonzero(as_tuple=False)

    if pred_points.shape[0] == 0 or gt_points.shape[0] == 0:
        return {f"{prefix}chamfer_{threshold}": torch.tensor(1e6)}

    gt_tree = cKDTree(gt_points.cpu().numpy())
    dist1, _ = gt_tree.query(pred_points.cpu().numpy(), k=1)
    pred_tree = cKDTree(pred_points.cpu().numpy())
    dist2, _ = pred_tree.query(gt_points.cpu().numpy(), k=1)
    chamfer_dist = torch.mean(torch.from_numpy(dist1**2)) + torch.mean(torch.from_numpy(dist2**2))
    
    return {
        f"{prefix}chamfer_{threshold}": chamfer_dist.item()
    }
