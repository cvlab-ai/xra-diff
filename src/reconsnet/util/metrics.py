import torch
import torch.nn.functional as F
import numpy as np

from scipy.ndimage import distance_transform_edt
from ..data.postprocess import denoise_voxels


def dice(pred, target, threshold=0.5, eps=1e-8):
    pred_bin = (pred > threshold).float()
    target_bin = (target > 0).float()

    intersection = (pred_bin * target_bin).sum()
    union = pred_bin.sum() + target_bin.sum()
    dice = (2.0 * intersection + eps) / (union + eps)
    return dice


def downsample(volume, tgt=(60,60,60)):
    _, _, d, h, w = volume.shape
    td, th, tw = tgt

    kd, kh, kw = d // td, h // th, w // tw
    sd, sh, sw = kd, kh, kw

    volume_down = F.max_pool3d(volume.float(), 
                               kernel_size=(kd, kh, kw),
                               stride=(sd, sh, sw),
                               ceil_mode=False)
    return volume_down


def add_tolerance(pred_bin, gt_bin, tol=1, convd=3):
        kernel_size = 2 * tol + 1
        if convd == 3:
            conv = F.conv3d
            kernelsz = (1, 1, kernel_size, kernel_size, kernel_size)
        else:
            conv = F.conv2d
            kernelsz = (1, 1, kernel_size, kernel_size)

        kernel = torch.ones(kernelsz, device=pred_bin.device)
        pred_dil = (conv(pred_bin, kernel, padding=tol) > 0).float()
        gt_dil   = (conv(gt_bin, kernel, padding=tol) > 0).float()
        return pred_dil, gt_dil


def confusion(pred, target, threshold, prefix="", suffix=None):
    if suffix is None: suffix = threshold
   
    pred_bin = (pred > threshold).float()
    pred_bin = denoise_voxels(pred_bin)
    gt_bin = (target > 0).float()
    
    pred_bin, gt_bin = add_tolerance(pred_bin, gt_bin)
    tp = (pred_bin * gt_bin).sum().item()
    fp = (pred_bin * (1 - gt_bin)).sum().item()
    fn = ((1 - pred_bin) * gt_bin).sum().item()
    tn = ((1 - pred_bin) * (1 - gt_bin)).sum().item()
    dice3d = (2 * tp) / (2 * tp + fp + fn + 1e-8)
    
    # print(pred.min(), pred.max(), pred.mean())
    # print("gt unique:", np.unique(target)[:10])
    # print("shapes:", pred.shape, target.shape)
        
    return {
        f"{prefix}dice3d_{suffix}": dice3d,
        f"{prefix}TP_{suffix}": tp, 
        f"{prefix}FP_{suffix}": fp, 
        f"{prefix}FN_{suffix}": fn, 
        f"{prefix}TN_{suffix}": tn
    }


def chamfer_distance(pred, target, threshold, prefix="", suffix=None, to_mm=1.0):
    if suffix is None: suffix = threshold

    pred_down = pred
    target_down = target
    
    pred_bin = (pred_down > threshold).float()
    pred_bin = denoise_voxels(pred_bin)
    gt_bin = (target_down > 0).float()
    
    pred_points = pred_bin.nonzero(as_tuple=False).float()
    gt_points = gt_bin.nonzero(as_tuple=False).float()
    
    if pred_points.shape[0] == 0 or gt_points.shape[0] == 0:
        return {f"{prefix}chamfer_{suffix}": 1e7}

    
    dist_matrix = torch.cdist(pred_points, gt_points, p=2)

    min_dist_1_to_2 = torch.min(dist_matrix, dim=1).values * to_mm
    min_dist_2_to_1 = torch.min(dist_matrix, dim=0).values * to_mm
    dist_1 = torch.mean(min_dist_1_to_2, dim=0)
    dist_2 = torch.mean(min_dist_2_to_1, dim=0)
    chamfer_dist = (dist_1 + dist_2) / 2.0

    return {
        f"{prefix}chamfer_{suffix}": chamfer_dist.item()
    }


def ot_metric(pred, target, threshold, d_mm, prefix="", suffix=None, to_mm=1.7):
    if suffix is None: suffix = threshold

    pred_down = pred
    target_down = target
    
    pred_bin = (pred_down > threshold).float()
    pred_bin = denoise_voxels(pred_bin)
    gt_bin = (target_down > 0).float()

    pred_points = pred_bin.nonzero(as_tuple=False).float()
    gt_points = gt_bin.nonzero(as_tuple=False).float()
    
    if pred_points.shape[0] == 0 or gt_points.shape[0] == 0:
        return {f"{prefix}Ot({d_mm})_{suffix}": 0}

    dist_matrix = torch.cdist(pred_points, gt_points, p=2) * to_mm

    tpr = (torch.min(dist_matrix, axis=0)[0] <= d_mm).sum()
    fn = dist_matrix.shape[1] - tpr
    tpm = (torch.min(dist_matrix, axis=1)[0] <= d_mm).sum()
    fp = dist_matrix.shape[0] - tpm

    ot_metric = (tpm + tpr) / (tpm + tpr + fp + fn)

    return {f"{prefix}Ot({d_mm})_{suffix}": float(ot_metric)}


def interpret_frac(pred, backproj, threshold=0.2):
    pred_bin = (pred >= threshold)
    backproj_bin = backproj.bool()
    
    intersection = (pred_bin & backproj_bin).sum().item()
    total_pred = pred_bin.sum().item()

    if total_pred == 0:
        return 0.0
    return intersection / total_pred


def chamfer_distance_image(img1, img2):
    img1 = img1.astype(bool)
    img2 = img2.astype(bool)

    dist_to_img2 = distance_transform_edt(~img2)
    dist_to_img1 = distance_transform_edt(~img1)

    d1 = dist_to_img2[img1].mean() if np.any(img1) else 0.0
    d2 = dist_to_img1[img2].mean() if np.any(img2) else 0.0
    return (d1 + d2)/2


def dice_image(img1, img2):
    img1 = torch.from_numpy(img1.astype(bool)).float().unsqueeze(dim=0)
    img2 = torch.from_numpy(img2.astype(bool)).float().unsqueeze(dim=0)
    img1, img2 = add_tolerance(img1, img2, tol=1, convd=2)
    
    intersection = torch.logical_and(img1, img2).sum()
    size1 = img1.sum()
    size2 = img2.sum()
    return 2.0 * intersection / (size1 + size2)
