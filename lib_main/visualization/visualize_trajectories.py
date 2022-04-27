import matplotlib.pyplot as plt

def visualize_trajectories(traj, ax=None):
    if ax is None:
        fig, ax = plt.subplot()

    if isinstance(traj, list):
        for trj_i in traj:
            ax.plot(trj_i[:,0], trj_i[:,1])
    else:
        ax.plot(traj[:, 0], traj[:, 1])

    return ax
