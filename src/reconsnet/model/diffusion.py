import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from diffusers import UNet3DConditionModel, DDPMScheduler, DDIMScheduler
from torchmetrics import PeakSignalNoiseRatio as PSNR

from ..config import get_config


class DiffusionModule(pl.LightningModule):
    def __init__(self, num_timesteps=1000, lr=1e-4):
        super().__init__()
        config = get_config()

        self.model = UNet3DConditionModel(
            sample_size=config['data']['grid_dim'],
            in_channels=1,
            out_channels=1,
            **config['model']
        )
        
        self.noise_scheduler = DDPMScheduler(
            **config['scheduler']
        )
        self.psnr = PSNR()
        self.lr = lr

    def forward(self, voxels, t, backprojection):
        return self.model(voxels, t, backprojection).sample
    
    def training_step(self, batch, _):
        x, y = batch
        noise = torch.randn_like(y)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (y.shape[0],), device=self.device).long()
    
        noisy_y = self.noise_scheduler.add_noise(y, noise, timesteps)
        noise_pred = self(noisy_y, timesteps, x)
        loss = F.mse_loss(noise_pred, noise)
        psnr_value = self.psnr(noise_pred, noise)
        self.log("loss", loss, on_epoch=True, on_step=True, prog_bar=True)
        self.log("psnr", psnr_value, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizers = [
            torch.optim.Adam(self.model.parameters(), lr=self.lr)
        ]
        return optimizers, []
