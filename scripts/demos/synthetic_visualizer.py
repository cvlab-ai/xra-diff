import argparse
import matplotlib
import matplotlib.pyplot as plt
import torch.nn.functional as F

from reconsnet.model.diffusion import DiffusionModule
from reconsnet.model.gan import GANModule
from reconsnet.model.unet3d import Unet3DModule
from reconsnet.data.dataset import XRayDatasetRight, XRayDatasetLeft
from reconsnet.util.visualize import visualize


parser = argparse.ArgumentParser()
parser.add_argument('--side', type=str, help='Side of the vessel', default="right")
parser.add_argument('--diffusion_checkpoint_path', type=str, help='Model weights path')
parser.add_argument('--gan_checkpoint_path', type=str, help='Model weights path')
parser.add_argument('--unet_checkpoint_path', type=str, help='Model weights path')
parser.add_argument('--sample_idx', type=int, help='Which sample from dataset to use as input data', default=0)
parser.add_argument('--dataset_path', type=str, help='ImageCAS dataset path', default="/home/shared/imagecas/projections_split/pilot")
parser.add_argument('--port', type=int, help='Where to serve the visualization', default=2137)
parser.add_argument('--sample_steps', type=int, help='How many steps to denoise with (only for DDIM)', default=50)
parser.add_argument('--no_guidance', action='store_true', help='disable cfg')
parser.add_argument('--slow', action='store_true', help='use slow sampling (DDPM)')
args = parser.parse_args()

diffusion_checkpoint_path = f"/home/shared/model-weights/{args.side}.ckpt" if args.diffusion_checkpoint_path is None else args.diffusion_checkpoint_path
gan_checkpoint_path = f"/home/shared/model-weights/baseline-gan-{args.side}.ckpt" if args.gan_checkpoint_path is None else args.gan_checkpoint_path
unet_checkpoint_path = f"/home/shared/model-weights/baseline-unet-{args.side}.ckpt" if args.unet_checkpoint_path is None else args.unet_checkpoint_path

matplotlib.use("WebAgg")
matplotlib.rcParams['webagg.port'] = args.port
matplotlib.rcParams['webagg.open_in_browser'] = False
ds = XRayDatasetRight(root_dir=args.dataset_path) if args.side == "right" else XRayDatasetLeft(root_dir=args.dataset_path)
model = DiffusionModule.load_from_checkpoint(diffusion_checkpoint_path, lr=1e-4)
gan = GANModule.load_from_checkpoint(gan_checkpoint_path)
unet = Unet3DModule.load_from_checkpoint(unet_checkpoint_path, lr=1e-5)

# sample = ds[args.sample_idx]

for sample in ds:
    diffusion_sample = (sample[0].to(model.device), sample[1].to(model.device), sample[2].to(model.device), sample[3].to(model.device))
    gan_sample = (sample[0].to(gan.device), sample[1].to(gan.device), sample[2].to(gan.device), sample[3].to(gan.device))
    unet_sample = (sample[0].to(unet.device), sample[1].to(unet.device), sample[2].to(unet.device), sample[3].to(unet.device))

    recons_fun = model.reconstruct if args.slow else lambda bp, p0, p1, guidance: model.fast_reconstruct(bp, p0, p1, args.sample_steps, guidance)
    gan.generator.to()

    _, s0 = visualize(
        lambda x: recons_fun(*x, not args.no_guidance),
        diffusion_sample
    )
    _, s1 = visualize(
        lambda x: gan.generator.forward(x[0]),
        gan_sample
    )
    _, s2 = visualize(
        lambda x: F.sigmoid(unet.forward(x[0])),
        unet_sample
    )
plt.show()
