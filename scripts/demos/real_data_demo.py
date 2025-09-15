import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from PIL import Image

from reconsnet.data.data import XRay
from reconsnet.data.preprocess import preprocess, xray_to_camera_model, ASSUMED_GRID_SPACING
from reconsnet.model.diffusion import DiffusionModule
from reconsnet.util.visualize import visualize
from reconsnet.config import get_config
from reconsnet.data.dataset import XRayDatasetRight
from reconsnet.util.coords import transpose
from reconsnet.util.camera import build_camera_model


XRAY0_PARAMS_PATH="data/params0.json"
XRAY0_IMAGE_PATH="data/xray0.png"
XRAY1_PARAMS_PATH="data/params1.json"
XRAY1_IMAGE_PATH="data/xray1.png"
CHECKPOINT_PATH="better-backproj-epoch200.ckpt"
SAMPLING_STEPS=10


def grid_to_pointcloud(grid, threshold=0.00):
    coords = np.argwhere(grid > threshold)
    return coords.T


def load_sample(params_path, image_path) -> XRay:
    with open(params_path, 'r') as f:
        params = json.load(f)
    
    img = np.array(Image.open(image_path).convert("F"))
    img = np.flipud(img).T.copy()
    print(img.shape)
    img_tensor = torch.tensor(img, dtype=torch.float32)
    
    sid = float(params['sid'])
    sod = float(params['sod'])
    alpha = np.deg2rad(float(params['alpha']))
    beta = np.deg2rad(float(params['beta']))
    spacing = float(params['spacing'])
    size = img.shape
    
    return XRay(img=img_tensor, sid=sid, sod=sod, alpha=alpha, beta=beta, spacing=spacing, size=size)


def make_confusion_overlay(pred, gt, threshold=0.6,
                           color_tp=(0, 255, 0),
                           color_fp=(255, 0, 0),
                           color_fn=(0, 0, 255),
                           color_tn=(0, 0, 0)):
    h, w = pred.shape
    overlay = np.zeros((h, w, 3), dtype=np.uint8)

    pred_bin = (pred > threshold).astype(np.uint8)
    gt_bin   = (gt   > threshold).astype(np.uint8)

    tp = (pred_bin == 1) & (gt_bin == 1)
    fp = (pred_bin == 1) & (gt_bin == 0)
    fn = (pred_bin == 0) & (gt_bin == 1)
    tn = (pred_bin == 0) & (gt_bin == 0)
    overlay[tp] = color_tp
    overlay[fp] = color_fp
    overlay[fn] = color_fn
    overlay[tn] = color_tn

    return overlay


def main():    
    camera_grid_size = [128, 128, 128]
    grid_dim = get_config()['data']['grid_dim']
    model = DiffusionModule.load_from_checkpoint(CHECKPOINT_PATH, lr=1e-4)
    xray0 = load_sample(XRAY0_PARAMS_PATH, XRAY0_IMAGE_PATH)
    xray1 = load_sample(XRAY1_PARAMS_PATH, XRAY1_IMAGE_PATH)
    
    backproj = preprocess(xray0, xray1, grid_size=camera_grid_size)
    backproj = (backproj - backproj.min()) / (backproj.max() - backproj.min() + 1e-8)
    backproj = torch.from_numpy(backproj).float().unsqueeze(0).unsqueeze(0)
    backproj = F.interpolate(backproj, size=(grid_dim, grid_dim, grid_dim), mode='trilinear', align_corners=False).squeeze(0)

    ds_right = XRayDatasetRight(root_dir="/home/shared/imagecas/projections")
    ds_sample = ds_right[0][0][0]
    
    backproj = backproj.to(model.device)
    pred_vox = visualize(lambda x: model.fast_reconstruct(x, SAMPLING_STEPS), (backproj, backproj))
    
    camera_grid_size = [60, 60, 60]
    trf0 = xray_to_camera_model(xray0, grid_size=camera_grid_size, grid_spacing=ASSUMED_GRID_SPACING * 128 / 60)
    trf1 = xray_to_camera_model(xray1, grid_size=camera_grid_size, grid_spacing=ASSUMED_GRID_SPACING * 128 / 60)
    
    pred0 = trf0(pred_vox)[0]
    pred1 = trf1(pred_vox)[0]
    
    _, axes = plt.subplots(2, 1)
 
    compare0 = make_confusion_overlay(np.array(pred0), xray0.img.numpy())
    compare1 = make_confusion_overlay(np.array(pred1), xray1.img.numpy())
 
    axes[0].imshow(compare0)
    axes[1].imshow(compare1)
    plt.show()
    
    


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("WebAgg")
    matplotlib.rcParams["webagg.port"] = 2137
    matplotlib.rcParams['webagg.open_in_browser'] = False
    main()
