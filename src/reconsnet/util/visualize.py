
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.widgets import Slider

from ..data.postprocess import denoise_pcd


def visualize(reconstruct, sample, initial_threshold=0.5, marker_size=2):
    backprojection, gt = sample

    pred = reconstruct(backprojection.unsqueeze(0)).squeeze()
    gt = gt.squeeze()
    pred_np_raw = pred.cpu().numpy()
    gt_np = (gt.squeeze() > initial_threshold).cpu().numpy()
    gt_coords = np.argwhere(gt_np)

    threshold = initial_threshold
    pred_np = (pred_np_raw > threshold)
    pred_coords = denoise_pcd(np.argwhere(pred_np))

    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(bottom=0.2)

    ax.scatter(gt_coords[:,0], gt_coords[:,1], gt_coords[:,2],
                         c='cyan', s=marker_size, alpha=0.7, label='GT')
    scat_pred = ax.scatter(pred_coords[:,0], pred_coords[:,1], pred_coords[:,2],
                           c='orange', s=marker_size, alpha=0.7, label='Prediction')

    ax.set_axis_off()
    ax.legend(loc='upper right')

    ax_slider = plt.axes([0.2, 0.05, 0.65, 0.03])
    slider = Slider(ax_slider, 'Threshold', 0.0, 1.0, valinit=initial_threshold)

    def update(val):
        th = slider.val
        new_coords = denoise_pcd(np.argwhere(pred_np_raw > th))

        scat_pred._offsets3d = (new_coords[:,0], new_coords[:,1], new_coords[:,2])
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()