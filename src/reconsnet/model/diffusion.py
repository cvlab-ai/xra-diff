import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

from diffusers import UNet3DConditionModel, DDPMScheduler, DDIMScheduler
from torchmetrics import PeakSignalNoiseRatio as PSNR
from tqdm import tqdm

from ..config import get_config
from ..util.metrics import dice


class DiffusionModule(pl.LightningModule):
    def __init__(self, lr, guidance_scale_override=None, take_projections=True):
        super().__init__()
        config = get_config()
        self.model = UNet3DConditionModel(
            sample_size=config['data']['grid_dim'],
            in_channels=2,
            out_channels=1,
            **config['unet3d']
        )
        self.noise_scheduler = DDPMScheduler(
            **config['scheduler']
        )
        self.bp_encoder = BackprojectionEncoder()
        self.p0_encoder = ProjectionEncoder()
        self.p1_encoder = ProjectionEncoder()
        if guidance_scale_override is None:
            self.guidance_scale = config['guidance']['scale']
        else:
            self.guidance_scale = guidance_scale_override
        self.drop_proba = config['guidance']['drop_proba']
        self.psnr = PSNR()
        self.lr = lr
        self.take_projections = take_projections

    def forward(self, voxels, t, backprojection, projection0, projection1, drop_proba=0.0):
        rand = torch.rand(1, device=self.device).item()
        cond = torch.zeros_like(backprojection, device=self.device) if rand < drop_proba else backprojection
        bp_feat = self.bp_encoder(cond)
        p0_feat = self.p0_encoder(torch.zeros_like(projection0, device=self.device) if rand < drop_proba else projection0)
        p1_feat = self.p1_encoder(torch.zeros_like(projection1, device=self.device) if rand < drop_proba else projection1)
        if self.take_projections:
            feat = bp_feat + p0_feat + p1_feat
        else:
            feat = bp_feat
        inp = torch.cat([voxels, cond], dim=1)
        return self.model(inp, t, feat).sample
    
    def guided_forward(self, voxels, t, backprojection, p0, p1):
        cond = self(voxels, t, backprojection, p0, p1, drop_proba=0.0)
        uncond = self(voxels, t, backprojection, p0, p1, drop_proba=1.0)
        return uncond + (cond - uncond) * self.guidance_scale

    def step(self, batch, log_prefix=""):
        backprojection, gt, p0, p1 = batch
        p0 = p0.to(self.device)
        p1 = p1.to(self.device)

        noise = torch.randn_like(gt, device=self.device)
        timesteps = torch.randint(
            0, 
            self.noise_scheduler.config.num_train_timesteps, (gt.shape[0],), 
            device=self.device
        ).long()
        noisy_y = self.noise_scheduler.add_noise(gt, noise, timesteps)
        noise_pred = self(noisy_y, timesteps, backprojection, p0, p1, drop_proba=self.drop_proba)
    
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
        x, y, p0, p1 = batch
        x = x.to(self.device)
        y = y.to(self.device)  
        p0 = p0.to(self.device)
        p1 = p1.to(self.device)
        sampled_voxels = self.reconstruct(x, p0, p1)

        def log_sample(ix, threshold=0.5):
            voxels = sampled_voxels[ix][0]
            gt_voxels = y[ix][0]
            backproj = x[ix][0]

            dice_gt_recons = dice(voxels, gt_voxels, threshold)
            dice_gt_backproj = dice(backproj, gt_voxels, threshold)
            dice_improvement = dice_gt_recons - dice_gt_backproj

            pcd = torch.argwhere(voxels > threshold).cpu().numpy()
            gt_pcd = torch.argwhere(gt_voxels > 0).cpu().numpy()
            gt_pcd = (gt_voxels > 0).nonzero(as_tuple=False).cpu().numpy()
            backproj_pcd = (backproj > 0).nonzero(as_tuple=False).cpu().numpy()

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
            
            self.log(f"Dice_{ix}_{threshold}_gt_recons", dice_gt_recons, on_epoch=True, prog_bar=True)
            self.log(f"Dice_{ix}_{threshold}_gt_backproj", dice_gt_backproj, on_epoch=True, prog_bar=True)
            self.log(f"Dice_{ix}_{threshold}_improvement", dice_improvement, on_epoch=True, prog_bar=True)
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
    def reconstruct(self, backprojection, p0, p1, guidance=True):
        self.eval()
        num_samples = backprojection.shape[0]
        device = self.device
        x = torch.randn_like(backprojection)
        timesteps = self.noise_scheduler.timesteps
        for t in tqdm(timesteps):
            t_tensor = torch.full((num_samples,), t, device=device, dtype=torch.long)
            if guidance:
                noise_pred = self.guided_forward(x, t_tensor, backprojection, p0, p1)
            else:
                noise_pred = self(x, t_tensor, backprojection, p0, p1)            
            x = self.noise_scheduler.step(noise_pred, t, x).prev_sample
        return x

    @torch.no_grad()
    def fast_reconstruct(self, backprojection, p0, p1, num_inference_steps=50, guidance=True):
        self.eval()
        num_samples = backprojection.shape[0]
        device = self.device
        x = torch.randn_like(backprojection)
        
        scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
            clip_sample=False,
        )
        scheduler.set_timesteps(num_inference_steps=num_inference_steps)         
        
        for t in tqdm(scheduler.timesteps):
            t_tensor = torch.full((num_samples,), t, device=device, dtype=torch.long)
            if guidance:
                noise_pred = self.guided_forward(x, t_tensor, backprojection, p0, p1)
            else:
                noise_pred = self(x, t_tensor, backprojection, p0, p1)
            x = scheduler.step(noise_pred, t, x).prev_sample
        return x


class BackprojectionEncoder(nn.Module):
    def __init__(self, in_channels=1, hidden_channels=64, out_features=1024):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(hidden_channels, out_features)
        self.norm1 = nn.BatchNorm3d(hidden_channels)
        self.norm2 = nn.BatchNorm3d(hidden_channels)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.activation(self.norm1(self.conv1(x)))
        x = self.activation(self.norm2(self.conv2(x)))
        x = self.conv3(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = x.unsqueeze(1)
        return x


class ProjectionEncoder(nn.Module):
    def __init__(self, in_channels=1, hidden_channels=64, out_features=1024):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1) 
        self.fc = nn.Linear(hidden_channels, out_features)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = x.unsqueeze(1)
        return x
