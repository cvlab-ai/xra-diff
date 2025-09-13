import pandas as pd
import time
import torch.nn.functional as F
import torch
import numpy as np

from tqdm import tqdm

from .metrics import dice, confusion
from ..config import get_config
from ..data.preprocess import preprocess
from .camera import build_camera_model


ASSUMED_GRID_SPACING = 0.8
THRESHOLD_RANGE = {
    "start": 0.1,
    "stop": 1.0,
    "num": 20
}

def synthetic_test(
    model, 
    ds, 
    csv_output_path,
):
    def make_test(reconstruct):
        df = []
        for i in tqdm(range(len(ds))):
            backprojection, gt = ds[i]
            backprojection = backprojection.to(model.device)
            gt = gt.to(model.device).unsqueeze(0) # add batch dim
            before = time.time()
            hat = reconstruct(backprojection.unsqueeze(0))
                        
            hat = (hat - hat.min()) / (hat.max() - hat.min())
            elapsed = time.time() - before
            entry = {
                "elapsed": elapsed
            }
            for threshold in np.linspace(**THRESHOLD_RANGE):
                entry = {
                    **entry,
                    **confusion(hat, gt, threshold, prefix="refined_")
                }
                entry = {
                    **entry,
                    **confusion(backprojection.unsqueeze(0), gt, threshold, prefix="backproj_")
                }   

                                   
            df.append(pd.DataFrame([entry]))
        return pd.concat(df)

    make_test(lambda x: model.fast_reconstruct(x, num_inference_steps=10, guidance=True)).to_csv(csv_output_path)


def clinical_test(
    model, 
    xray_pairs,
    csv_ddpm_output_path,
    csv_ddim_output_path
):
    grid_dim = get_config()['data']['grid_dim']
    
    def xray_to_camera_model(xray):
        return build_camera_model(
            xray.alpha,
            xray.beta,
            xray.sid,
            xray.sod,
            ASSUMED_GRID_SPACING,
            [128, 128, 128],
            xray.spacing,
            xray.size
    )    

    def make_test(reconstruct):
        df = pd.DataFrame()
        for (xray0, xray1) in tqdm(xray_pairs):
            before = time.time()
            
            preprocessed = preprocess(
                xray0,
                xray1,
                [128, 128, 128]
            )

            preprocessed = (preprocessed - preprocessed.min()) / (preprocessed.max() - preprocessed.min() + 1e-8)
            preprocessed = torch.from_numpy(preprocessed).float().unsqueeze(0).unsqueeze(0)
            preprocessed = F.interpolate(preprocessed, size=(grid_dim, grid_dim, grid_dim), mode='trilinear', align_corners=False)

            rec = reconstruct(preprocessed)
            camera0 = xray_to_camera_model(xray0)
            camera1 = xray_to_camera_model(xray1)
            
            projection0 = camera0(rec)[0]
            projection1 = camera1(rec)[0]

            # TODO: compute reconstruction errors (projection vs xray)
            # recd0 = dice(projection0, xray0.image)
            # recd1 = dice(projection1, xray1.image)
            
            elapsed = time.time() - before
            df.add({
                "elapsed": elapsed,
                # "recd0": recd0,
                # "recd1": recd1
            })

        return df
    
    make_test(model.reconstruct).to_csv(csv_ddpm_output_path)
    make_test(model.fast_reconstruct).to_csv(csv_ddim_output_path)
