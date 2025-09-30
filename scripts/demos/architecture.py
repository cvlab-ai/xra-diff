import torch
import torch.nn as nn
import pandas as pd

from pytorch_lightning.utilities.model_summary import ModelSummary
from reconsnet.model.diffusion import DiffusionModule
from reconsnet.config import get_config

CHECKPOINT_PATH="stronger-conditioning.ckpt"
config = get_config()

grid_dim = config["data"]["grid_dim"]
batch_size=2

model = DiffusionModule.load_from_checkpoint(CHECKPOINT_PATH, lr=1e-4)
ms = ModelSummary(model, max_depth=2)
print(ms)
