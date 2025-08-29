import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from torch import autograd
from torch.autograd import Variable

from diffusers import UNet3DConditionModel, DDPMScheduler, DDIMScheduler
from torchmetrics import PeakSignalNoiseRatio as PSNR
from tqdm import tqdm

from .blocks.generator import Generator
from .blocks.discriminator import Discriminator
from ..config import get_config
from ..util.metrics import dice



class GANModule(pl.LightningModule):
    def __init__(self, lr=1e-4):
        super().__init__()
        config = get_config()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.lr = lr
        self.generator = Generator(in_channels=1, num_filters=64, class_num=1)

        # device is not set to cuda yet
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.discriminator = Discriminator(device, 2)

    def training_step(self, batch):
        volumes, gt = batch

        optimizer_g, optimizer_d = self.optimizers()
        optimizer_g.zero_grad()
        optimizer_d.zero_grad()

        output = self.generator(volumes)

        DX_score = self.discriminator(torch.cat((volumes, gt), 1)).mean() # D(x)
        DG_score = self.discriminator(torch.cat((volumes, output), 1).detach()).mean()

        gradient_penalty = self._calculate_gradient_penalty(
            torch.cat((volumes, gt), 1), 
            torch.cat((volumes, output), 1).detach(),
            self.discriminator,
            self.device,
            volumes.shape[0]
        )

        d_loss = (DG_score - DX_score + gradient_penalty)
        Wasserstein_D = DX_score - DG_score

        self.log(f"train_discriminator_loss", d_loss, on_epoch=True, on_step=True, prog_bar=True)
        self.log(f"train_Wasserstein_loss", Wasserstein_D, on_epoch=True, on_step=True, prog_bar=True)
        
        self.toggle_optimizer(optimizer_d)
        self.manual_backward(d_loss)
        optimizer_d.step()
        self.untoggle_optimizer(optimizer_d)

        output = self.generator(volumes)
        
        DG_score = self.discriminator(torch.cat((volumes, output), 1).detach()).mean()
        G_loss = -DG_score
        l1_loss = self._generation_eval(output, gt)
        combined_loss = G_loss + l1_loss*100

        self.log(f"train_generator_simple_loss", G_loss, on_epoch=True, on_step=True, prog_bar=True)
        self.log(f"train_generator_l1_loss", l1_loss, on_epoch=True, on_step=True, prog_bar=True)
        self.log(f"train_generator_combined_loss", combined_loss, on_epoch=True, on_step=True, prog_bar=True)

        self.toggle_optimizer(optimizer_g)
        self.manual_backward(combined_loss)
        optimizer_g.step()
        self.untoggle_optimizer(optimizer_g)

    def validation_step(self, batch, batch_idx):
        volumes, gt = batch
        
        outputs = self.generator(volumes)

        DG_score = self.discriminator(torch.cat((volumes, outputs), 1)).mean() # D(G(z))
        G_loss = -DG_score
        l1_loss = self._generation_eval(outputs, gt)
        combined_loss = G_loss + l1_loss * 100

        self.log(f"val_generator_simple_loss", G_loss, on_epoch=True, on_step=True, prog_bar=True)
        self.log(f"val_generator_l1_loss", l1_loss, on_epoch=True, on_step=True, prog_bar=True)
        self.log(f"val_generator_combined_loss", combined_loss, on_epoch=True, on_step=True, prog_bar=True)
    
    def on_validation_epoch_end(self):
        # if self.current_epoch % 10: return
        val_loader = self.trainer.datamodule.val_dataloader()
        batch = next(iter(val_loader))
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)  
        sampled_voxels = self.generator(x)

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
        g_optimizer = optim.Adam(
            self.generator.parameters(),
            lr=self.lr,
            betas=(0.5, 0.9)
        )
        d_optimizer = optim.Adam(
            self.discriminator.parameters(), 
            lr=self.lr,
            betas=(0.5, 0.9)
        )
        return g_optimizer, d_optimizer

    def _calculate_gradient_penalty(self, real_images, fake_images, discriminator, device, batch_size):
        eta = torch.FloatTensor(batch_size,2,1,1,1).uniform_(0,1).to(device)
        eta = eta.expand(batch_size, real_images.size(1), real_images.size(2), real_images.size(3), real_images.size(4))

        interpolated = eta * fake_images + ((1 - eta) * real_images)
        interpolated = Variable(interpolated, requires_grad=True)
        prob_interpolated = discriminator(interpolated)

        gradients = autograd.grad(
            outputs=prob_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones(
                prob_interpolated.size()
            ).to(device),
            create_graph=True, 
            retain_graph=True
        )[0]

        lambda_term = 10
        grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_term
        return grad_penalty

    def _generation_eval(self, outputs, labels):
        l1_criterion = nn.L1Loss() #nn.MSELoss()

        l1_loss = l1_criterion(outputs, labels)

        return l1_loss