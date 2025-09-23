import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch.nn.functional as F


from reconsnet.data.preprocess import preprocess, xray_to_camera_model, ASSUMED_GRID_SPACING
from reconsnet.model.diffusion import DiffusionModule
from reconsnet.model.gan import GANModule
from reconsnet.model.unet3d import Unet3DModule
from reconsnet.util.visualize import visualize, make_confusion_overlay
from reconsnet.config import get_config
from reconsnet.data.dataset import load_clinical_sample, default_transform
from reconsnet.util.coords import reproject

XRAY0_IMAGE_PATH="data/xray0.png"
XRAY1_IMAGE_PATH="data/xray1.png"
CHECKPOINT_PATH="stronger-conditioning.ckpt"
BASELINE_CHECKPOINT_PATH="baseline.ckpt"
UNET_CHECKPOINT_PATH="unet3d-baseline.ckpt"
SAMPLING_STEPS=10


parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', type=str, help='Model weights path', default="stronger-conditioning.ckpt")
parser.add_argument('--baseline_checkpoint_path', type=str, help='Model weights path', default="baseline.ckpt")
parser.add_argument('--unet_checkpoint_path', type=str, help='Model weights path', default="unet3d-baseline.ckpt")
parser.add_argument('--xray0_params_path', type=str, help="Parameters json for the first xray", default="data/params0.json")
parser.add_argument('--xray1_params_path', type=str, help="Parameters json for the second xray", default="data/params1.json")
parser.add_argument('--xray0_image_path', type=str, help="Image with the first xray", default="data/xray0.png")
parser.add_argument('--xray1_image_path', type=str, help="Image with the second xray", default="data/xray1.png")
parser.add_argument('--port', type=int, help='Where to serve the visualization', default=2137)
parser.add_argument('--sample_steps', type=int, help='How many steps to denoise with', default=10)
parser.add_argument('--camera_grid_size', type=int, help='How many steps to denoise with', default=60)

args = parser.parse_args()

matplotlib.use("WebAgg")
matplotlib.rcParams["webagg.port"] = args.port
matplotlib.rcParams['webagg.open_in_browser'] = False

camera_grid_size = [args.camera_grid_size] * 3
model = DiffusionModule.load_from_checkpoint(args.checkpoint_path, lr=1e-4)
baseline_model = GANModule.load_from_checkpoint(args.baseline_checkpoint_path)
unet_model = Unet3DModule.load_from_checkpoint(args.unet_checkpoint_path, lr=1e-5)
xray0 = load_clinical_sample(args.xray0_params_path, args.xray0_image_path)
xray1 = load_clinical_sample(args.xray1_params_path, args.xray1_image_path)

backproj, _, img0, img1 = default_transform([xray0, xray1], None)

backproj = backproj.to(model.device)
img0 = xray0.img.to(model.device).unsqueeze(0)
img1 = xray0.img.to(model.device).unsqueeze(0)


sample = (backproj, backproj, img0, img1)

pred_vox, slider0 = visualize(lambda x: model.fast_reconstruct(*x, num_inference_steps=SAMPLING_STEPS), sample, gt_label="backprojection")
basel_vox, slider1 = visualize(lambda x: baseline_model.generator.forward(x[0]), sample, gt_label="backprojection")
unet_vox, slider2 = visualize(lambda x: unet_model.forward(x[0]), sample, gt_label="backprojection")
pred0, pred1 = reproject(pred_vox, xray0, xray1, camera_grid_size)
basel_pred0, basel_pred1 = reproject(basel_vox, xray0, xray1, camera_grid_size)
unet_pred0, unet_pred1 = reproject(unet_vox, xray0, xray1, camera_grid_size)

_, axes = plt.subplots(6, 1)

compare0 = make_confusion_overlay(np.array(pred0), xray0.img.numpy())
compare1 = make_confusion_overlay(np.array(pred1), xray1.img.numpy())
basel_compare0 = make_confusion_overlay(np.array(basel_pred0), xray0.img.numpy())
basel_compare1 = make_confusion_overlay(np.array(basel_pred1), xray1.img.numpy())
unet_compare0 = make_confusion_overlay(np.array(unet_pred0), xray0.img.numpy())
unet_compare1 = make_confusion_overlay(np.array(unet_pred1), xray1.img.numpy())

axes[0].imshow(compare0)
axes[1].imshow(compare1)
axes[2].imshow(basel_compare0)
axes[3].imshow(basel_compare1)
axes[4].imshow(unet_compare0)
axes[5].imshow(unet_compare1)

plt.show()
