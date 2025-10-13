import pandas as pd
import time
import torch.nn.functional as F
import torch
import numpy as np

from tqdm import tqdm

from .metrics import *
from ..data.postprocess import percentile_threshold, denoise_voxels
from ..util.coords import reproject
from torchmetrics import PeakSignalNoiseRatio as PSNR


ASSUMED_GRID_SPACING = 0.8
THRESHOLD_RANGE = {
    "start": 0.0,
    "stop": 1.0,
    "num": 50
}
SAVE_EVERY=10

@torch.no_grad()
def synthetic_test(
    model, 
    ds, 
    csv_output_path,
    reconstruct,
    repeat_each=3
):
    psnr = PSNR().to(model.device)
    def make_test(reconstruct):
        df = []
        for i in tqdm(range(len(ds))):
            for _ in range(repeat_each):
                backprojection, gt, p0, p1 = ds[i]
                backprojection = backprojection.to(model.device)
                p0 = p0.to(model.device)
                p1 = p1.to(model.device)
                gt = gt.to(model.device).unsqueeze(0) # add batch dim
                before = time.time()
                hat = reconstruct((backprojection.unsqueeze(0), p0.unsqueeze(0), p1.unsqueeze(0)))
                            
                hat = (hat - hat.min()) / (hat.max() - hat.min())
                elapsed = time.time() - before
                entry = {
                    "elapsed": elapsed
                }
                
                backprojection = backprojection.unsqueeze(0)
                
                for threshold in np.linspace(**THRESHOLD_RANGE):
                    entry = {
                        **entry,
                        **confusion(hat, gt, threshold, prefix="refined_"),
                        **chamfer_distance(hat, gt, threshold, prefix="refined_"),
                        **ot_metric(hat, gt, threshold, d_mm=0, prefix="refined_"),
                        **ot_metric(hat, gt, threshold, d_mm=1, prefix="refined_"),
                        **ot_metric(hat, gt, threshold, d_mm=2, prefix="refined_"),
                        **confusion(hat, gt, threshold, prefix="backproj_"),
                        **chamfer_distance(backprojection, gt, threshold, prefix="backproj_"),
                        **ot_metric(backprojection, gt, threshold, d_mm=0, prefix="backproj_"),
                        **ot_metric(backprojection, gt, threshold, d_mm=1, prefix="backproj_"),
                        **ot_metric(backprojection, gt, threshold, d_mm=2, prefix="backproj_")
                    }
                entry = {
                    **entry,
                    "interpret_frac": interpret_frac(hat, backprojection),
                    "PSNR": psnr(hat, gt).item()
                }
                  
                
                df.append(pd.DataFrame([entry]))
            if i % SAVE_EVERY == 0: pd.concat(df).to_csv(csv_output_path)
        return pd.concat(df)
    make_test(reconstruct).to_csv(csv_output_path)


@torch.no_grad()
def synthetic_test_adaptive(
    model, 
    ds, 
    csv_output_path,
    reconstruct,
    repeat_each=3,
    camera_grid_size = [60] * 3,
):
    psnr = PSNR().to(model.device)
    ds.return_projections = True
    def make_test(reconstruct):
        df = []
        for i in tqdm(range(len(ds))):
            for _ in range(repeat_each):
                (backprojection, gt, p0, p1), (xray0, xray1) = ds[i]
                backprojection = backprojection.to(model.device)
                p0 = p0.to(model.device)
                p1 = p1.to(model.device)
                gt = gt.to(model.device).unsqueeze(0) # add batch dim
                before = time.time()
                hat = reconstruct((backprojection.unsqueeze(0), p0.unsqueeze(0), p1.unsqueeze(0)))
                            
                hat = (hat - hat.min()) / (hat.max() - hat.min())
                elapsed = time.time() - before
                entry = {
                    "elapsed": elapsed
                }
                
                backprojection = backprojection.unsqueeze(0)
                threshold = percentile_threshold(hat)
                
                hat_bin = denoise_voxels((hat > threshold).float()).squeeze().cpu().numpy()
                pred0, pred1 = reproject(hat_bin, xray0, xray1, camera_grid_size)
                
                # API adaptation
                xray0.img = torch.from_numpy(xray0.img.asarray())
                xray1.img = torch.from_numpy(xray1.img.asarray())
            
                args = (pred0, pred1, pred0, pred1, xray0, xray1)

                entry = {
                    **entry,
                    **confusion(hat, gt, threshold, prefix="refined_", suffix="adaptive"),
                    **chamfer_distance(hat, gt, threshold, prefix="refined_", suffix="adaptive"),
                    **ot_metric(hat, gt, threshold, d_mm=0, prefix="refined_", suffix="adaptive"),
                    **ot_metric(hat, gt, threshold, d_mm=1, prefix="refined_", suffix="adaptive"),
                    **ot_metric(hat, gt, threshold, d_mm=2, prefix="refined_", suffix="adaptive"),
                    **confusion(backprojection, gt, threshold, prefix="backproj_", suffix="adaptive"),
                    **chamfer_distance(backprojection, gt, threshold, prefix="backproj_", suffix="adaptive"),
                    **ot_metric(backprojection, gt, threshold, d_mm=0, prefix="backproj_", suffix="adaptive"),
                    **ot_metric(backprojection, gt, threshold, d_mm=1, prefix="backproj_", suffix="adaptive"),
                    **ot_metric(backprojection, gt, threshold, d_mm=2, prefix="backproj_", suffix="adaptive"),
                    "interpret_frac": interpret_frac(hat, backprojection, threshold),
                    "PSNR": psnr(hat, gt).item(),
                    **earth_movers_distance(hat, gt, threshold, prefix="refined_", suffix="adaptive"),
                    **earth_movers_distance(backprojection, gt, threshold, prefix="refined_", suffix="adaptive"),
                    **_reproj_metric(chamfer_distance_image, "chamfer_distance", args),
                    **_reproj_metric(dice_image, "dice2d", args)
                }
                
                df.append(pd.DataFrame([entry]))
            if i % SAVE_EVERY == 0: pd.concat(df).to_csv(csv_output_path)
        return pd.concat(df)
    make_test(reconstruct).to_csv(csv_output_path)


def clinical_test(
    model, 
    ds, 
    csv_output_path,
    reconstruct,
    repeat_each=3,
    camera_grid_size = [60] * 3,
):
    def make_test(reconstruct):
        df = []
        for i in tqdm(range(len(ds))):
            for _ in range(repeat_each):
                (backprojection, _, p0, p1), xray0, xray1 = ds[i]
                backprojection = backprojection.to(model.device)
                p0 = p0.to(model.device)
                p1 = p1.to(model.device)
                before = time.time()
                hat = reconstruct((backprojection.unsqueeze(0), p0.unsqueeze(0).unsqueeze(0), p1.unsqueeze(0).unsqueeze(0)))
                hat = (hat - hat.min()) / (hat.max() - hat.min())
                elapsed = time.time() - before
                entry = {
                    "elapsed": elapsed
                }
                threshold = percentile_threshold(hat)
                hat_bin = denoise_voxels((hat > threshold).float()).squeeze().cpu().numpy()
                bp_bin = (backprojection > 0).squeeze().cpu().numpy()

                pred0, pred1 = reproject(hat_bin, xray0, xray1, camera_grid_size)
                backproj0, backproj1 = reproject(bp_bin, xray0, xray1, camera_grid_size)
                args = (pred0, pred1, backproj0, backproj1, xray0, xray1)

                entry = {
                    **_reproj_metric(chamfer_distance_image, "chamfer_distance", args),
                    **_reproj_metric(dice_image, "dice2d", args)
                }
              
                df.append(pd.DataFrame([entry]))
            if i % SAVE_EVERY == 0: pd.concat(df).to_csv(csv_output_path)
        return pd.concat(df)
    make_test(reconstruct).to_csv(csv_output_path)


def _reproj_metric(f, name, args):
    pred0, pred1, backproj0, backproj1, xray0, xray1 = args
    
    m0 = f(pred0.asarray(), xray0.img.cpu().numpy()).item()
    m1 = f(pred1.asarray(), xray1.img.cpu().numpy()).item()
    bp_m0 = f(backproj0.asarray(), xray0.img.cpu().numpy()).item()
    bp_m1 = f(backproj1.asarray(), xray1.img.cpu().numpy()).item()
    return {
        f"{name}0": m0,
        f"{name}1": m1,
        f"bp_{name}0": bp_m0,
        f"bp_{name}1": bp_m1,
        f"{name}0_mm": m0 * xray0.spacing,
        f"{name}1_mm": m1 * xray1.spacing,
        f"bp_{name}0_mm": bp_m0 * xray0.spacing,
        f"bp_{name}1_mm": bp_m1 * xray1.spacing,
    }
                    
