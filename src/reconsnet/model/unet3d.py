import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

from monai.networks.nets.unet import UNet
from monai.networks.layers import Norm
from monai.losses.dice import DiceLoss

from ..config import get_config
from ..util.coords import pad_pow2, crop_grid
from ..util.metrics import dice


class Unet3DModule(pl.LightningModule):
    def __init__(self, lr):
        super().__init__()
        config = get_config()
        channels = config["unet3d"]["block_out_channels"]
        self.model =  UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            channels=channels,
            strides=[2] * (len(channels) - 1),
            num_res_units=2,
            norm=Norm.BATCH,
        )
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([100.0]))
        self.lr = lr

    def forward(self, backprojection):
        return crop_grid(self.model(pad_pow2(backprojection)))

    def step(self, batch, log_prefix=""):
        backprojection, gt, _, _ = batch
        output = self(backprojection)
        
        loss = self.loss_fn(output, gt)
        self.log(f"{log_prefix}loss", loss, on_epoch=True, on_step=True, prog_bar=True)
        
        return loss

    def training_step(self, batch, _):
        return self.step(batch, log_prefix="train_")
    
    def validation_step(self, batch, _):
        return self.step(batch, log_prefix="val_")

    def on_validation_epoch_end(self):
        if self.current_epoch % 10: return
        val_loader = self.trainer.datamodule.val_dataloader()
        batch = next(iter(val_loader))
        x, y, _, _ = batch
        x = x.to(self.device)
        y = y.to(self.device)  
        sampled_voxels = F.sigmoid(self(x))

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

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer
