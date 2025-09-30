import yaml
import torch
import torch.nn.functional as F

from pathlib import Path
from random import uniform

from reconsnet.config import get_config
from reconsnet.util.test import synthetic_test_adaptive
from reconsnet.model.diffusion import DiffusionModule
from reconsnet.data.dataset import XRayDatasetRight, default_transform
from reconsnet.util.camera import build_camera_model


CHECKPOINT_PATH = "stronger-conditioning.ckpt"
DATA_PATH = "/home/shared/imagecas/projections_split/val"
RESULTS_PATH = "data/synthetic_extra_projections_right.csv"
MODEL = DiffusionModule.load_from_checkpoint(CHECKPOINT_PATH, lr=1e-4)
RECONSTRUCT = lambda x: MODEL.fast_reconstruct(*x, num_inference_steps=10, guidance=True)
CONFIG_PATH=Path("config")
with open(CONFIG_PATH / "projections.yaml") as f:
    CONFIG = yaml.SafeLoader(f.read()).get_data()


def inject_extra_projection(projections, gt, inject_second=False):
    global_config = CONFIG['global']
    img_res = global_config['image_resolution']
    grid_res = global_config['grid_resolution']
    grid_spacing = uniform(*global_config["grid_spacing"]) 
    grid_dim = get_config()['data']['grid_dim']

    xray0 = projections[0]
    bp_before, gt_down, img0, img1 = default_transform(projections, gt)
    
    def perpendicular_bp(xray):
        alpha = -xray.beta
        beta = xray.alpha
        sid = xray.sid
        sod = xray.sod    
        img_spacing = xray.spacing
        camera = build_camera_model(alpha, beta, sid, sod, grid_spacing, grid_res, img_spacing, img_res)
        img = camera(gt)[0]
        bp = camera.adjoint(img).asarray()
              
        bp = (bp - bp.min()) / (bp.max() - bp.min() + 1e-8)
        bp = torch.from_numpy(bp).float().unsqueeze(0).unsqueeze(0)
        bp = F.interpolate(bp, size=(grid_dim, grid_dim, grid_dim), mode='trilinear', align_corners=False)
        return bp.squeeze(0)
     
    bp_after = torch.minimum(perpendicular_bp(xray0), bp_before)
    if inject_second:
        xray1 = projections[1]
        bp_after = torch.minimum(perpendicular_bp(xray1), bp_after)
    
    return bp_after, gt_down, img0, img1

# P = 3
synthetic_test_adaptive(
    model=MODEL,
    ds=XRayDatasetRight(root_dir=DATA_PATH, transform=inject_extra_projection),
    csv_output_path=RESULTS_PATH,
    reconstruct=RECONSTRUCT
)

# P = 4
synthetic_test_adaptive(
    model=MODEL,
    ds=XRayDatasetRight(root_dir=DATA_PATH, transform=lambda p, gt: inject_extra_projection(p, gt, True)),
    csv_output_path=RESULTS_PATH,
    reconstruct=RECONSTRUCT
)
