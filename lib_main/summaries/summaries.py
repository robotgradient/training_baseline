import numpy as np
import torchvision
import torch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt


class SummaryDict():
    def __init__(self, summaries):
        self.fields = summaries.keys()
        self.summaries = summaries

    def compute_summary(self, model, model_input, ground_truth, info , writer, iter, prefix=""):
        for field in self.fields:
            prefix_in = prefix + field
            self.summaries[field](model, model_input, ground_truth, info[field], writer, iter, prefix_in)


def ebm_summary(model, model_input, ground_truth, info , writer, iter, prefix=""):
    """Writes tensorboard summaries using tensorboardx api.

    :param writer: tensorboardx writer object.
    :param predictions: Output of forward pass.
    :param ground_truth: Ground truth.
    :param iter: Iteration number.
    :param prefix: Every summary will be prefixed with this string.
    """
    pred_energy = info['model_outputs']['ene'][...,None]
    gt_energy = info['labels'][...,None]

    writer.add_scalar(prefix + "out_min", pred_energy.min(), iter)
    writer.add_scalar(prefix + "out_max", pred_energy.max(), iter)

    writer.add_scalar(prefix + "trgt_min", gt_energy.min(), iter)
    writer.add_scalar(prefix + "trgt_max", gt_energy.max(), iter)


    ## Evaluate under random coordinate values ##
    inputs = info['model_inputs']
    inputs['x_ene'] = inputs['x_occ']
    pred_energy = model(inputs, field = 'ene')
    coords = info['model_inputs']['x_ene']
    pred_energy = pred_energy['ene'][..., None]


    input_coords = coords[:1].detach().cpu().numpy()
    pred_occ_coords = coords[:1, pred_energy[0].squeeze(-1) > 0.5, :].detach().cpu().numpy()
    color = np.zeros_like(pred_occ_coords)
    color[...,0] = np.ones_like(color[...,0])*250


    ## Add Table ##
    p = info['model_inputs']['point_cloud'][None, 0, :300, ...]
    p = p.detach().cpu().numpy()
    pred_occ_coords = np.concatenate((pred_occ_coords, p), 1)

    p_color = np.zeros_like(p)
    p_color[...,0] = np.zeros_like(p_color[...,0])
    color = np.concatenate((color, p_color), 1)

    point_cloud(writer, iter, prefix+'_predicted_coords', pred_occ_coords, colors=color)


def occupancy_summary(model, model_input, ground_truth, info, writer, iter, prefix=""):
    """Writes tensorboard summaries using tensorboardx api.

    :param writer: tensorboardx writer object.
    :param predictions: Output of forward pass.
    :param ground_truth: Ground truth.
    :param iter: Iteration number.
    :param prefix: Every summary will be prefixed with this string.
    """
    pred_occ = info['occ']['occ'][:, :, None]
    gt_occ = ground_truth['occ']
    coords = model_input['x_occ']

    ## Occupancy Max-Min ##
    writer.add_scalar(prefix + "out_min", pred_occ.min(), iter)
    writer.add_scalar(prefix + "out_max", pred_occ.max(), iter)

    # writer.add_scalar(prefix + "trgt_min", gt_occ.min(), iter)
    # writer.add_scalar(prefix + "trgt_max", gt_occ.max(), iter)

    ## Set colors based on good occupancy predictions ##
    input_coords = coords[:1].detach().cpu().numpy()


    corr_mask = (gt_occ[0].squeeze(-1) > 0) == (pred_occ[0].squeeze(-1) > 0.5)
    all_colors = np.zeros_like(input_coords)
    all_colors[0, corr_mask.detach().cpu().numpy()] = np.array([[200., 0., 0.]])

    ## Matches ##
    matches = (corr_mask==False).sum()
    writer.add_scalar(prefix + "matches", matches, iter)

    # input_coords = coords[:1].detach().cpu().numpy()
    # gt_occ_coords = coords[:1, gt_occ[0].squeeze(-1) > 0.5, :].detach().cpu().numpy()
    # pred_occ_coords = coords[:1, pred_occ[0].squeeze(-1) > 0.5, :].detach().cpu().numpy()
    #
    # # Compute colors for point predictions
    # all_colors = np.ones_like(input_coords)
    # all_colors[:, :, 1:] = 0.
    # corr_mask = (gt_occ[0].squeeze(-1) > 0) == (pred_occ[0].squeeze(-1) > 0.5)
    # all_colors[0, corr_mask.detach().cpu().numpy()] = np.array([[0., 200., 0.]])
    # pred_occ_colors = all_colors[:1, pred_occ[0].squeeze(-1).detach().cpu().numpy() > 0]

    # point_cloud(writer, iter, 'input_coords', input_coords)
    # point_cloud(writer, iter, prefix+'_ground_truth_coords', gt_occ_coords)
    # point_cloud(writer, iter, prefix+'_predicted_coords', pred_occ_coords, colors=pred_occ_colors)

    point_cloud(writer, iter, prefix+'_predicted_coords', input_coords, colors=all_colors)



def point_cloud(writer, iter, name, points_xyz, colors=None):
    point_size_config = {
        'material': {
            'cls': 'PointsMaterial',
            'size': 0.05
        }
    }

    if colors is None:
       colors = np.zeros_like(points_xyz)

    writer.add_mesh(name, vertices=points_xyz, colors=colors,
                     config_dict={"material": point_size_config}, global_step=iter)


def get_occ_ebm_summary():
    summaries = {'occ': occupancy_summary, 'ene':ebm_summary}
    summary_dict = SummaryDict(summaries=summaries)
    return summary_dict.compute_summary