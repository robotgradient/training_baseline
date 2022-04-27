import numpy as np
import torchvision
import torch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt

from .utils import point_cloud


def sdf_summary(model, model_input, ground_truth, info, writer, iter, prefix=""):
    """Writes tensorboard summaries using tensorboardx api.

    :param writer: tensorboardx writer object.
    :param predictions: Output of forward pass.
    :param ground_truth: Ground truth.
    :param iter: Iteration number.
    :param prefix: Every summary will be prefixed with this string.
    """
    pred_sdf = info['sdf']['sdf'][:, :, None]
    gt_sdf = ground_truth['sdf']
    coords = model_input['x_sdf']

    ## Occupancy Max-Min ##
    writer.add_scalar(prefix + "out_min", pred_sdf.min(), iter)
    writer.add_scalar(prefix + "out_max", pred_sdf.max(), iter)

    ## Set colors based on good occupancy predictions ##
    input_coords = coords[:1].detach().cpu().numpy()
    pred_sdf = pred_sdf[:1,...].detach().cpu().numpy()

    all_colors = np.zeros_like(input_coords)
    def set_color(all_colors, l_thrs=0., h_thrs=0.1, i=1, intensity=200):
        idxs = np.argwhere((pred_sdf<h_thrs) & (pred_sdf>l_thrs))[:,1]
        color = np.zeros((1,3))
        color[0,i] = intensity

        all_colors[:, idxs,...] = color
        return all_colors
    all_colors = set_color(all_colors, h_thrs=0.1, i=0)
    all_colors = set_color(all_colors, l_thrs=0.1, h_thrs=0.3, i=1)
    all_colors = set_color(all_colors, l_thrs=0.3, h_thrs=0.5, i=2)


    point_cloud(writer, iter, prefix+'_colorized_sdf', input_coords, colors=all_colors)

