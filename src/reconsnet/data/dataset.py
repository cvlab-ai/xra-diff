import torch
import pytorch_lightning as pl
import joblib
import torch.nn.functional as F
import json
import numpy as np

from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split
from .preprocess import preprocess
from ..config import get_config
from .data import XRay
from PIL import Image
from scipy.ndimage import shift, zoom


def default_transform(projections, gt):
    grid_dim = get_config()['data']['grid_dim']
    preprocessed = preprocess(
        projections[0],
        projections[1],
        [128, 128, 128] # OG grid
    )

    if gt is not None:
        gt = torch.from_numpy(gt).float().unsqueeze(0).unsqueeze(0)
        gt = F.interpolate(gt, size=(grid_dim, grid_dim, grid_dim), mode='trilinear', align_corners=False)
        gt = gt.squeeze(0)
        
    preprocessed = (preprocessed - preprocessed.min()) / (preprocessed.max() - preprocessed.min() + 1e-8)
    preprocessed = torch.from_numpy(preprocessed).float().unsqueeze(0).unsqueeze(0)
    preprocessed = F.interpolate(preprocessed, size=(grid_dim, grid_dim, grid_dim), mode='trilinear', align_corners=False)
    
    def xray_to_tensor(x):
        if isinstance(x.img, torch.Tensor):
            tens = x.img
        else:
            tens = torch.from_numpy(x.img.asarray()).unsqueeze(0)

        return (tens - tens.min()) / (tens.max() - tens.min() + 1e-8)
        
    return preprocessed.squeeze(0), gt, xray_to_tensor(projections[0]), xray_to_tensor(projections[1])

class XRayDataset(Dataset):
    def __init__(self, root_dir, side, transform=default_transform, return_projections=False):
        self.root = Path(root_dir)
        self.transform = transform
        self.side = side
        self.paths = list(self.root.glob(f"{side}*.joblib"))
        self.return_projections = return_projections
    
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        loaded = joblib.load(path)
        projections = loaded["projections"]
        gt = loaded["gt"]
        trfed = self.transform(projections, gt)
        
        if self.return_projections:
            return trfed, projections
        return trfed

class XRayDatasetBoth(XRayDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, side="")


class XRayDatasetLeft(XRayDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, side="left")
        

class XRayDatasetRight(XRayDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, side="right")


class XRayDataModule(pl.LightningDataModule):
    def __init__(self, train_dataset, val_dataset, batch_size, num_workers, val_split, **_):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split


    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          )

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          )        
        

def _get_dm(train_root_dir, val_root_dir, Dataset):
    return XRayDataModule(
        Dataset(root_dir=train_root_dir),
        Dataset(root_dir=val_root_dir),
        **get_config()['data']
    )
    

def get_dm_left(train_root_dir, val_root_dir):
    return _get_dm(train_root_dir, val_root_dir, XRayDatasetLeft)


def get_dm_right(train_root_dir, val_root_dir):
    return _get_dm(train_root_dir, val_root_dir, XRayDatasetRight)


def get_dm_both(train_root_dir, val_root_dir):
    return _get_dm(train_root_dir, val_root_dir, XRayDatasetBoth)


def load_clinical_sample(params_path, image_path, dicom=False) -> XRay:
    import ast
    
    with open(params_path, 'r') as f:
        params = json.load(f)
    
    img = np.array(Image.open(image_path).convert("F"))
    img = np.flipud(img).T.copy()
    img_tensor = torch.tensor(img, dtype=torch.float32)
    
    if dicom:
        sid = float(params["(0018,1110)"]["value"])
        sod = float(params["(0018,1111)"]["value"])
        alpha = np.deg2rad(float(params['(0018,1510)']["value"]))
        beta = np.deg2rad(float(params['(0018,1511)']["value"]))
        spacing = float(ast.literal_eval(params["(0018,1164)"]["value"])[0])
    else:
        sid = float(params['sid'])
        sod = float(params['sod'])
        alpha = np.deg2rad(float(params['alpha']))
        beta = np.deg2rad(float(params['beta']))
        spacing = float(params['spacing'])
    size = img.shape
    return XRay(img=img_tensor, sid=sid, sod=sod, alpha=alpha, beta=beta, spacing=spacing, size=size)


def random_move(img, translation_range):
    arr = img.asarray()
    ndim = arr.ndim
    shifts = np.random.uniform(-translation_range, translation_range, size=ndim)
    moved = shift(arr, shift=shifts, mode="reflect")
    return img.space.element(moved)


def random_scale(img, scaling_range):
    arr = img.asarray()
    factor = np.random.uniform(1 - scaling_range, 1 + scaling_range)
    scaled = zoom(arr, zoom=factor, order=1, mode="reflect")

    out = np.zeros_like(arr)
    in_shape  = np.array(scaled.shape)
    out_shape = np.array(arr.shape)

    slices_in  = []
    slices_out = []
    for s_in, s_out in zip(in_shape, out_shape):
        if s_in >= s_out:      # crop center
            start_in = (s_in - s_out) // 2
            slices_in.append(slice(start_in, start_in + s_out))
            slices_out.append(slice(None))
        else:                  # pad center
            start_out = (s_out - s_in) // 2
            slices_in.append(slice(None))
            slices_out.append(slice(start_out, start_out + s_in))

    out[tuple(slices_out)] = scaled[tuple(slices_in)]
    return img.space.element(out)


class ClinicalDataset(Dataset):
    def __init__(self, root_dir):
        """
        Args:
            root_dir (str or Path): folder containing params*.json and mask*.png
            transform (callable, optional): transform for the image/mask pair
            mask_transform (callable, optional): transform just for mask if different
        """
        self.root = Path(root_dir)
        self.pairs = []
        
        for sub in sorted(self.root.iterdir(), key=lambda p: p.name):
            self.pairs.append(
                (
                    { "params": sub / "params0.json", "mask": sub / "mask0.png"},
                    { "params": sub / "params1.json", "mask": sub / "mask1.png"},
                )
            )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        sample0, sample1 = self.pairs[idx]
        xray0 = load_clinical_sample(sample0["params"], sample0["mask"], True)
        xray1 = load_clinical_sample(sample1["params"], sample1["mask"], True)
        return default_transform([xray0, xray1], None), xray0, xray1
