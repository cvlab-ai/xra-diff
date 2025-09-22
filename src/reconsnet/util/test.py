import pandas as pd
import time
import torch.nn.functional as F
import torch
import numpy as np

from tqdm import tqdm

from .metrics import confusion, chamfer_distance, interpret_frac, downsample
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
                hat_down = downsample(hat, tgt=(30, 30, 30))
                gt_down = downsample(gt, tgt=(30, 30, 30))
                bp_down = downsample(backprojection, tgt=(30, 30, 30))
                
                for threshold in np.linspace(**THRESHOLD_RANGE):
                    entry = {
                        **entry,
                        **confusion(hat_down, gt_down, threshold, prefix="refined_"),
                        **chamfer_distance(hat, gt, threshold, prefix="refined_"),
                        **confusion(bp_down, gt_down, threshold, prefix="backproj_"),
                        **chamfer_distance(backprojection, gt, threshold, prefix="backproj_"),
                   
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
                hat_down = downsample(hat, tgt=(30, 30, 30))
                gt_down = downsample(gt, tgt=(30, 30, 30))
                bp_down = downsample(backprojection, tgt=(30, 30, 30))
                
                threshold = percentile_threshold(hat)
                threshold_down = percentile_threshold(hat_down)
                entry = {
                    **entry,
                    **confusion(hat_down, gt_down, threshold_down, prefix="refined_", suffix="adaptive"),
                    **chamfer_distance(hat, gt, threshold, prefix="refined_", suffix="adaptive"),
                    **confusion(bp_down, gt_down, threshold_down, prefix="backproj_", suffix="adaptive"),
                    **chamfer_distance(backprojection, gt, threshold, prefix="backproj_", suffix="adaptive"),
                    "interpret_frac": interpret_frac(hat, backprojection, threshold),
                    "PSNR": psnr(hat, gt)
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
    camera_grid_size = [60] * 3
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

                backprojection = backprojection.unsqueeze(0)
                pred0, pred1 = reproject(hat_bin, xray0, xray1, camera_grid_size)
                # TODO: add metrics
                df.append(pd.DataFrame([entry]))
            if i % SAVE_EVERY == 0: pd.concat(df).to_csv(csv_output_path)
        return pd.concat(df)
    make_test(reconstruct).to_csv(csv_output_path)
