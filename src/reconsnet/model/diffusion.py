import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
from diffusers import UNet3DConditionModel, DDPMScheduler, DDIMScheduler
from torchmetrics import PeakSignalNoiseRatio as PSNR
from tqdm import tqdm

from ..config import get_config


class DiffusionModule(pl.LightningModule):
    def __init__(self, lr):
        super().__init__()
        config = get_config()
        self.model = UNet3DConditionModel(
            sample_size=config['data']['grid_dim'],
            in_channels=2,
            out_channels=1,
            **config['model']
        )
        self.noise_scheduler = DDPMScheduler(
            **config['scheduler']
        )
        self.psnr = PSNR()
        self.lr = lr

    def forward(self, voxels, t, backprojection):
        dummy = torch.zeros((voxels.shape[0], 1, 1024), device=self.device)
        inp = torch.cat([voxels, backprojection], dim=1)
        return self.model(inp, t, dummy).sample
    
    def step(self, batch, log_prefix=""):
        x, y = batch
        assert x.dim()==5 and y.dim()==5 and x.shape==y.shape, f"Got x {x.shape}, y {y.shape}"

        noise = torch.randn_like(y)
        timesteps = torch.randint(
            0, 
            self.noise_scheduler.config.num_train_timesteps, (y.shape[0],), 
            device=self.device
        ).long()
        noisy_y = self.noise_scheduler.add_noise(y, noise, timesteps)
        noise_pred = self(noisy_y, timesteps, x)
    
        loss = F.mse_loss(noise_pred, noise)
        psnr_value = self.psnr(noise_pred, noise)
        self.log(f"{log_prefix}loss", loss, on_epoch=True, on_step=True, prog_bar=True)
        self.log(f"{log_prefix}psnr", psnr_value, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def training_step(self, batch, _):
        return self.step(batch, log_prefix="train_")
    
    def validation_step(self, batch, _):
        return self.step(batch, log_prefix="val_")

    def configure_optimizers(self):
        optimizers = [
            torch.optim.Adam(self.model.parameters(), lr=self.lr)
        ]
        return optimizers, []

    def on_validation_epoch_end(self):
        if self.current_epoch % 10: return
        val_loader = self.trainer.datamodule.val_dataloader()
        batch = next(iter(val_loader))
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)  
        sampled_voxels = self.reconstruct(x)

        def log_sample(ix, threshold=0.5):
            voxels = sampled_voxels[ix][0]
            gt_voxels = y[ix][0]
            backproj = x[ix][0]

            def log_pcd(pcd, title):
                fig_gen = plt.figure(figsize=(8, 8))
                ax_gen = fig_gen.add_subplot(111, projection='3d')
                ax_gen.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], s=1, c='blue', alpha=0.3)
                ax_gen.set_title(title)
                ax_gen.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], s=1, c='red')
                ax_gen.set_xlabel('X')
                ax_gen.set_ylabel('Y')
                ax_gen.set_zlabel('Z')
                ax_gen.set_box_aspect([1,1,1])
                fig_gen.canvas.draw()
                img_gen = np.array(fig_gen.canvas.renderer.buffer_rgba())
                self.logger.experiment.add_image(title, img_gen.transpose(2, 0, 1), self.current_epoch)
                plt.close(fig_gen)
            
            pcd = torch.argwhere(voxels > threshold).cpu().numpy()
            gt_pcd = torch.argwhere(gt_voxels > 0).cpu().numpy()
            
            gt_pcd = (gt_voxels > 0).nonzero(as_tuple=False).cpu().numpy()
            backproj_pcd = (backproj > 0).nonzero(as_tuple=False).cpu().numpy()
            log_pcd(pcd, f"Reconstructed_{ix}_{threshold} PCD Plot")
            log_pcd(gt_pcd, f"GT_{ix}")
            log_pcd(backproj_pcd, f"Backprojected_{ix}")

        batch_size = x.shape[0]
        for ix, thr in [
            (0, 0.5),
            (1, 0.5),
            (2, 0.5),
            (0, 0.9),
            (1, 0.9),
            (2, 0.9)
        ]: 
            if ix >= batch_size: continue
            log_sample(ix, thr)

    @torch.no_grad
    def reconstruct(self, backprojection):
        self.eval()
        num_samples = backprojection.shape[0]
        device = self.device
        x = torch.randn_like(backprojection)
        timesteps = self.noise_scheduler.timesteps
        for t in tqdm(timesteps):
            t_tensor = torch.full((num_samples,), t, device=device, dtype=torch.long)
            noise_pred = self(x, t_tensor, backprojection)
            x = self.noise_scheduler.step(noise_pred, t, x).prev_sample
        return x

