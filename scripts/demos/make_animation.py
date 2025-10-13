import torch
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import numpy as np

from tqdm import tqdm
from diffusers import DDIMScheduler

from reconsnet.data.postprocess import denoise_voxels, percentile_threshold
from reconsnet.data.dataset import ClinicalDataset
from reconsnet.model.diffusion import DiffusionModule


CAMERA_GRID_SIZE = (60, 60, 60)
STEPS = 50
SLOWDOWN = 9
POWER=2


@torch.no_grad()
def animate(model, backprojection, p0, p1, num_inference_steps=STEPS):
    model.eval()
    num_samples = backprojection.shape[0]
    device = model.device
    x = torch.randn_like(backprojection)

    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
        clip_sample=False,
    )
    scheduler.set_timesteps(num_inference_steps=num_inference_steps)         

    frames = []

    def frames_in_t(t):
        progress = t /num_inference_steps
        return 1 + int(SLOWDOWN * progress**POWER)

    for i, t in enumerate(tqdm(scheduler.timesteps, desc="Making the animation...")):
        t_tensor = torch.full((num_samples,), t, device=device, dtype=torch.long)
        noise_pred = model.guided_forward(x, t_tensor, backprojection, p0, p1)
        x = scheduler.step(noise_pred, t, x).prev_sample
        threshold = percentile_threshold(x)
        x_bin = denoise_voxels((x > threshold).float()).squeeze().cpu().numpy()
        x_coords = np.argwhere(x_bin)
        frcnt = frames_in_t(i)
        for _ in range(frcnt): frames.append(x_coords)


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    graph = ax.scatter(frames[-1][:, 0], frames[-1][:, 1], frames[-1][:, 2], s=5,  c='black', alpha=1.0)
    ax.axis('off')

    def update(frame):
        graph._offsets3d = (frame[:, 0], frame[:, 1], frame[:, 2])
        return graph,
    return FuncAnimation(fig, update, frames=frames, interval=40, blit=False)


def make_animation(model, ds, ix, name):
    (bp, _, p0, p1), _, _ = ds[ix]
    bp = bp.to(model.device).unsqueeze(0)
    p0 = p0.to(model.device).unsqueeze(0).unsqueeze(0)
    p1 = p1.to(model.device).unsqueeze(0).unsqueeze(0)
    anim = animate(model, bp, p0, p1)
    anim.save(f"media/{name}")


if __name__ == "__main__":        
    model_right = DiffusionModule.load_from_checkpoint("/home/shared/model-weights/both.ckpt", lr=1e-4)
    model_left = DiffusionModule.load_from_checkpoint("/home/shared/model-weights/left-noproj.ckpt", lr=1e-4)

    ds_right = ClinicalDataset("/home/shared/uck-right")
    ds_left = ClinicalDataset("/home/shared/uck-left")
    
    make_animation(model_right, ds_right, 0, "animation0.gif")
    make_animation(model_right, ds_right, 1, "animation1.gif")
    make_animation(model_right, ds_right, 2, "animation2.gif")
    make_animation(model_left, ds_left, 0, "animation3.gif")
    make_animation(model_left, ds_left, 1, "animation4.gif")
    make_animation(model_left, ds_left, 2, "animation5.gif")
