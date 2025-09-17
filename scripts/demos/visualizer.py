import argparse
import matplotlib
import matplotlib.pyplot as plt

from reconsnet.model.diffusion import DiffusionModule
from reconsnet.model.gan import GANModule
from reconsnet.data.dataset import XRayDatasetRight
from reconsnet.util.visualize import visualize


parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', type=str, help='Model weights path', default="vanilla-200.ckpt")
parser.add_argument('--baseline_checkpoint_path', type=str, help='Model weights path', default="baseline.ckpt")
parser.add_argument('--sample_idx', type=int, help='Which sample from dataset to use as input data', default=0)
parser.add_argument('--dataset_path', type=str, help='ImageCAS dataset path', default="/home/shared/imagecas/projections_split/val")
parser.add_argument('--port', type=int, help='Where to serve the visualization', default=2137)
parser.add_argument('--sample_steps', type=int, help='How many steps to denoise with (only for DDIM)', default=50)
parser.add_argument('--no_guidance', action='store_true', help='disable cfg')
parser.add_argument('--slow', action='store_true', help='use slow sampling (DDPM)')
args = parser.parse_args()

matplotlib.use("WebAgg")
matplotlib.rcParams['webagg.port'] = args.port
matplotlib.rcParams['webagg.open_in_browser'] = False
ds = XRayDatasetRight(root_dir=args.dataset_path)
model = DiffusionModule.load_from_checkpoint(args.checkpoint_path, lr=1e-4)
baseline = GANModule.load_from_checkpoint(args.baseline_checkpoint_path)

sample = ds[args.sample_idx]
sample = (sample[0].to(model.device), sample[1].to(model.device), sample[2].to(model.device), sample[3].to(model.device))

recons_fun = model.reconstruct if args.slow else lambda bp, p0, p1, guidance: model.fast_reconstruct(bp, p0, p1, args.sample_steps, guidance)

_, s0 = visualize(
    lambda x: recons_fun(*x, not args.no_guidance),
    sample
)
_, s1 = visualize(
    lambda x: baseline.generator.forward(x[0]),
    sample
)
plt.show()
