import torch
import pytorch_lightning as pl
import joblib
import torch.nn.functional as F

from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split
from .preprocess import preprocess
from ..config import get_config


def default_transform(projections, gt):
    grid_dim = get_config()['data']['grid_dim']
    preprocessed = preprocess(
        projections[0],
        projections[1],
        [grid_dim, grid_dim, grid_dim]
    )

    gt = torch.from_numpy(gt).float().unsqueeze(0).unsqueeze(0)
    gt = F.interpolate(gt, size=(grid_dim, grid_dim, grid_dim), mode='trilinear', align_corners=False)
    gt = gt.squeeze(0)

    preprocessed = torch.from_numpy(preprocessed).float()
    preprocessed = (preprocessed - preprocessed.min()) / (preprocessed.max() - preprocessed.min() + 1e-8)
    return preprocessed.unsqueeze(0), gt

class XRayDataset(Dataset):
    def __init__(self, root_dir, side, transform=default_transform):
        self.root = Path(root_dir)
        self.transform = transform
        self.side = side
        self.paths = list(self.root.glob(f"{side}*.joblib"))
    
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        loaded = joblib.load(path)
        projections = loaded["projections"]
        gt = loaded["gt"]
        return self.transform(projections, gt)


class XRayDatasetLeft(XRayDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, side="left")
        

class XRayDatasetRight(XRayDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, side="right")


class XRayDataModule(pl.LightningDataModule):
    def __init__(self, dataset, batch_size, num_workers, val_split, **_):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.setup()

    def setup(self, stage=None):
            val_size = int(len(self.dataset) * self.val_split)
            train_size = len(self.dataset) - val_size
            self.train_dataset, self.val_dataset = random_split(self.dataset, [train_size, val_size])
            
            print(f"Initialized dataset for {self.dataset.side} arteries consisting of {len(self.dataset)} samples ({len(self.train_dataset)} train, {len(self.val_dataset)} val)")
            

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          )

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          )        
        

def _get_dm(root_dir, Dataset):
    return XRayDataModule(
        Dataset(root_dir=root_dir),
        **get_config()['data']
    )
    

def get_dm_left(root_dir):
    return _get_dm(root_dir, XRayDatasetLeft)


def get_dm_right(root_dir):
    return _get_dm(root_dir, XRayDatasetRight)
