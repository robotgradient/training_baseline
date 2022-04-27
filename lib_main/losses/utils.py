from lib_main import samplers, losses


def get_occ_and_ene_losses(device='cpu', neg_min = -2., neg_max = 2.):
    # Loss Energy
    dim = 3
    NegativeSampler = samplers.Uniform(lows=[neg_min]*dim, highs= [neg_max]*dim, dim=dim, device=device)
    loss = losses.InfoNCE(negative_sampler=NegativeSampler)
    loss_fn_ene = loss.loss_fn

    # Loss Occupancy
    loss = losses.OccupancyLoss()
    loss_fn_occ = loss.loss_fn

    # Loss Dictionary
    loss_fns = {'occ':loss_fn_occ, 'ene':loss_fn_ene}
    loss_dict = losses.LossDictionary(loss_dict=loss_fns)
    return loss_dict


def get_occ_and_se3_ene_losses(device='cpu', neg_min = -2., neg_max = 2.):
    # Loss Energy
    dim = 3
    NegativeSampler = samplers.SE3_Uniform(t_lows=[neg_min]*dim, t_highs= [neg_max]*dim, device=device)
    loss = losses.InfoNCE(negative_sampler=NegativeSampler)
    loss_fn_ene = loss.loss_fn

    # Loss Occupancy
    loss = losses.OccupancyLoss()
    loss_fn_occ = loss.loss_fn

    # Loss Dictionary
    #loss_fns = {'occ':loss_fn_occ, 'ene':loss_fn_ene}
    loss_fns = {'ene':loss_fn_ene}
    loss_dict = losses.LossDictionary(loss_dict=loss_fns)
    return loss_dict