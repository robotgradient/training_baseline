import numpy as np
import torch
import trimesh


def create_gripper_marker(color=[0, 0, 255], tube_radius=0.001, sections=6, scale = 1.):
    """Create a 3D mesh visualizing a parallel yaw gripper. It consists of four cylinders.

    Args:
        color (list, optional): RGB values of marker. Defaults to [0, 0, 255].
        tube_radius (float, optional): Radius of cylinders. Defaults to 0.001.
        sections (int, optional): Number of sections of each cylinder. Defaults to 6.

    Returns:
        trimesh.Trimesh: A mesh that represents a simple parallel yaw gripper.
    """
    cfl = trimesh.creation.cylinder(
        radius=0.002*scale,
        sections=sections,
        segment=[
            [4.10000000e-02*scale, -7.27595772e-12*scale, 6.59999996e-02*scale],
            [4.10000000e-02*scale, -7.27595772e-12*scale, 1.12169998e-01*scale],
        ],
    )
    cfr = trimesh.creation.cylinder(
        radius=0.002*scale,
        sections=sections,
        segment=[
            [-4.100000e-02*scale, -7.27595772e-12*scale, 6.59999996e-02*scale],
            [-4.100000e-02*scale, -7.27595772e-12*scale, 1.12169998e-01*scale],
        ],
    )
    cb1 = trimesh.creation.cylinder(
        radius=0.002*scale, sections=sections, segment=[[0, 0, 0], [0, 0, 6.59999996e-02*scale]]
    )
    cb2 = trimesh.creation.cylinder(
        radius=0.002*scale,
        sections=sections,
        segment=[[-4.100000e-02*scale, 0, 6.59999996e-02*scale], [4.100000e-02*scale, 0, 6.59999996e-02*scale]],
    )

    tmp = trimesh.util.concatenate([cb1, cb2, cfr, cfl])
    tmp.visual.face_colors = color

    return tmp


def visualize_points(model, input):

    scale = input['scale'].cpu().numpy()
    model.eval()
    with torch.no_grad():
        x, occ = model.get_points_and_features(input)

    ## Visualization points ##
    point_clouds = input['point_cloud'].cpu().numpy()
    x = x.cpu().numpy()
    occ = occ.cpu().numpy()
    H = input['x_ene'].cpu().numpy()

    ## Trimesh
    p_cloud_tri = trimesh.points.PointCloud(point_clouds[0,...])

    pc2 = trimesh.points.PointCloud(x[0,0,...], r=10)
    colors = np.zeros((x[0,0,...].shape[0],3))

    colors[:,0] = (occ[0,0,...]<0.25)*1.
    colors[:,1] = (occ[0,0,...]>0.25)*1.
    pc2.colors = colors


    print('max occ: ',occ.max())
    print('min occ: ', occ.min())

    grip = create_gripper_marker(scale=scale[0,0]).apply_transform(H[0,0,...])
    trimesh.Scene([p_cloud_tri, pc2, grip]).show()
