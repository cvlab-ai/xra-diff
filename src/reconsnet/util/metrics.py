import torch


def dice(pred, target, threshold=0.5, eps=1e-8):
    pred_bin = (pred > threshold).float()
    target_bin = (target > 0).float()

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
