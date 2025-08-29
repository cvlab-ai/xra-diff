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




class GANModule(pl.LightningModule):
    def __init__(self, lr=1e-4):
        super().__init__()
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
        
        self.toggle_optimizer(optimizer_d)
        self.manual_backward(d_loss)
        optimizer_d.step()
        self.untoggle_optimizer(optimizer_d)

        output = self.generator(volumes)
        
        DG_score = self.discriminator(torch.cat((volumes, output), 1).detach()).mean()
        G_loss = -DG_score
        l1_loss = self._generation_eval(output, gt)
        combined_loss = G_loss + l1_loss*100

        self.toggle_optimizer(optimizer_g)
        self.manual_backward(combined_loss)
        optimizer_g.step()
        self.untoggle_optimizer(optimizer_g)

    def validation_step(self, batch, batch_idx):
        pass

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