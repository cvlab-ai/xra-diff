
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.widgets import Slider

from ..data.postprocess import denoise_voxels, percentile_threshold
from ..util.metrics import confusion, chamfer_distance


def visualize(reconstruct, sample, initial_threshold=0.5, marker_size=2):
    backprojection, gt, p0, p1 = sample

    pred = reconstruct((backprojection.unsqueeze(0), p0.unsqueeze(0), p1.unsqueeze(0)))
    gt = gt.squeeze()
    gt_np = (gt.squeeze() > initial_threshold).cpu().numpy()
    gt_coords = np.argwhere(gt_np)
    bp_coords = np.argwhere((backprojection > 0.2).squeeze().cpu().numpy())

    threshold = percentile_threshold(pred)
    
    pred_bin = denoise_voxels((pred > threshold).float()).squeeze().cpu().numpy()
    pred_coords = np.argwhere(pred_bin)

    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(bottom=0.2)

    ax.scatter(gt_coords[:,0], gt_coords[:,1], gt_coords[:,2],
                         c='cyan', s=marker_size, alpha=0.7, label='GT')
    ax.scatter(bp_coords[:, 0], bp_coords[:, 1], bp_coords[:, 2], alpha=0.5)
    scat_pred = ax.scatter(pred_coords[:,0], pred_coords[:,1], pred_coords[:,2],
                           c='orange', s=marker_size, alpha=0.7, label='Prediction')

    ax.set_axis_off()
    ax.legend(loc='upper right')

    ax_slider = plt.axes([0.2, 0.05, 0.65, 0.03])
    slider = Slider(ax_slider, 'Threshold', 0.0, 1.0, valinit=initial_threshold)

    def update(val):
        th = slider.val
        pred_bin_tensor = denoise_voxels((pred > th).float()).squeeze() 
        pred_bin = pred_bin_tensor.cpu().numpy()
        new_coords = np.argwhere(pred_bin)

        scat_pred._offsets3d = (new_coords[:,0], new_coords[:,1], new_coords[:,2])
        fig.canvas.draw_idle()
        
        print(
            confusion(pred, gt.unsqueeze(0).unsqueeze(0), th),
            chamfer_distance(pred, gt.unsqueeze(0).unsqueeze(0), th)
        )
    

    slider.on_changed(update)
    return pred_bin, slider
