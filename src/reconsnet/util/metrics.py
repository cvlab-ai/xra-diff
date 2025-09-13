import torch
import torch.nn.functional as F


from ..data.postprocess import denoise_voxels


def dice(pred, target, threshold=0.5, eps=1e-8):
    pred_bin = (pred > threshold).float()
    target_bin = (target > 0).float()

    print("SUMS", pred_bin.sum(), target_bin.sum())

    intersection = (pred_bin * target_bin).sum()
    union = pred_bin.sum() + target_bin.sum()
    dice = (2.0 * intersection + eps) / (union + eps)
    return dice


def cd(pred, target, threshold):
    pred_points = torch.nonzero(pred > threshold, as_tuple=False).float()
    target_points = torch.nonzero(target > 0, as_tuple=False).float()
    
    if len(pred_points) == 0 or len(target_points) == 0:
        return torch.tensor(float("inf"))
    
    dist_matrix = torch.cdist(pred_points, target_points, p=2) 
    return dist_matrix.min(dim=1)[0].mean() + dist_matrix.min(dim=0)[0].mean()

def confusion(pred, target, threshold, prefix=""):
    def downsample(volume, tgt=(60,60,60)):
        volume_down = F.interpolate(volume.float(), size=tgt, mode="trilinear", align_corners=False)
        return volume_down
    
    pred_down = downsample(pred)
    target_down = downsample(target)
    
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
        # f"{prefix}TP_{threshold}": tp, 
        # f"{prefix}FP_{threshold}": fp, 
        # f"{prefix}FN_{threshold}": fn, 
        # f"{prefix}TN_{threshold}": tn
    }
