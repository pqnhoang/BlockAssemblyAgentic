import numpy as np
import genesis as gs
import time
from scipy.spatial.transform import Rotation
import cv2
import os
from datetime import datetime
from PIL import Image
import argparse

dt = 5e-4
E, nu = 3.e4, 0.45
rho = 1000.

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    args = parser.parse_args()

    ########################## init ##########################
    gs.init(backend=gs.cpu)

    ########################## create a scene ##########################

    scene = gs.Scene(
    viewer_options = gs.options.ViewerOptions(
        camera_pos    = (0, -2.5, 1.5),
        camera_lookat = (0.0, 0.0, 0.5),
        camera_fov    = 30,
        max_FPS       = 60,
    ),
    sim_options = gs.options.SimOptions(
        dt = 0.01,
    ),
    show_viewer=args.vis,
    rigid_options=gs.options.RigidOptions(
        gravity=(0.0, 0.0, -10.0),
    ),
    mpm_options=gs.options.MPMOptions(
        dt=dt,
        lower_bound=(-1.0, -1.0, -0.2),
        upper_bound=( 1.0,  1.0,  1.0),
    ),
    fem_options=gs.options.FEMOptions(
        dt=dt,
        damping=45.,
    ),
    vis_options=gs.options.VisOptions(
        show_world_frame=False,
    ),
)

    ########################## entities ##########################
    plane = scene.add_entity(
        gs.morphs.Plane(),
    )
    
    # Add table leg (cylinder)
    table_leg = scene.add_entity(
        gs.morphs.Cylinder(
            radius=0.02,  # 20mm radius
            height=0.04,  # 40mm height
            pos=(0.25, 0.1, 0.02),  # Initial position
        ),
        material=gs.materials.Rigid(
            friction=1,
        ),
        surface=gs.surfaces.Default(
            color=(0.5, 0.5, 0.5),
        )
    )

    worm = scene.add_entity(
        morph=gs.morphs.Mesh(
            file='meshes/worm/worm.obj',
            pos=(0.3, 0.3, 0.001),
            scale=0.1,
            euler=(90, 0, 0),
        ),
        material=gs.materials.MPM.Muscle(
            E=5e5,
            nu=0.45,
            rho=10000.,
            model='neohooken',
            n_groups=4,
        ),
    )

    def set_muscle_by_pos(robot):
        if isinstance(robot.material, gs.materials.MPM.Muscle):
            pos = robot.get_state().pos
            n_units = robot.n_particles
        elif isinstance(robot.material, gs.materials.FEM.Muscle):
            pos = robot.get_state().pos[robot.get_el2v()].mean(1)
            n_units = robot.n_elements
        else:
            raise NotImplementedError

        pos = pos.cpu().numpy()
        pos_max, pos_min = pos.max(0), pos.min(0)
        pos_range = pos_max - pos_min

        lu_thresh, fh_thresh = 0.3, 0.6
        muscle_group = np.zeros((n_units,), dtype=int)
        mask_upper = pos[:, 2] > (pos_min[2] + pos_range[2] * lu_thresh)
        mask_fore = pos[:, 1] < (pos_min[1] + pos_range[1] * fh_thresh)
        muscle_group[ mask_upper &  mask_fore] = 0 # upper fore body
        muscle_group[ mask_upper & ~mask_fore] = 1 # upper hind body
        muscle_group[~mask_upper &  mask_fore] = 2 # lower fore body
        muscle_group[~mask_upper & ~mask_fore] = 3 # lower hind body

        muscle_direction = np.array([[0, 1, 0]] * n_units, dtype=float)

        robot.set_muscle(
            muscle_group=muscle_group,
            muscle_direction=muscle_direction,
        )

    
    # Define your Euler angles in degrees (roll, pitch, yaw) = (0, 0, 90)
    euler_angles = [0, 0, 90]  # in degrees

    # Convert Euler angles to quaternion
    # Note: scipy expects radians for the from_euler function
    r = Rotation.from_euler('xyz', [np.radians(angle) for angle in euler_angles])
    quat = r.as_quat()  # Returns in [x, y, z, w] format

    # # Adjust camera to match scene viewer
    # cam_top_to_bot = scene.add_camera(
    #     res=(1280, 720),
    #     pos=(0, 0, 3),  # Move camera back and up
    #     lookat=(0.0, 0.0, 0),  # Focus on center of structure
    #     fov=30,  # Slightly wider field of view
    #     GUI=False,
    # )
    # cam_left_to_right = scene.add_camera(
    #     res=(1280, 720),
    #     pos=(0, 3, 0),  # Move camera back and up
    #     lookat=(0.0, 0.0, 0),  # Focus on center of structure
    #     fov=30,  # Slightly wider field of view
    #     GUI=False,
    # )
    cam = scene.add_camera(
        res=(1280, 720),
        pos=(0, -2.5, 1.5),
        lookat=(0.0, 0.0, 0.5),
        fov=30,
        GUI=False,
    )
    ########################## build ##########################
    scene.build()
    ############ Optional: set control gains ############
    table_leg.set_quat(quat)
    set_muscle_by_pos(worm)
    scene.step()

    import sys
    if sys.platform == "darwin":
        scene._visualizer._viewer._pyrender_viewer._renderer.dpscale = 1

    gs.tools.run_in_another_thread(fn=run_sim, args=(scene, args.vis, cam,  worm))
    if args.vis:
        scene.viewer.start()


def run_sim(scene, enable_vis, cam, worm):
    cam.start_recording()
    
    for i in range(1000):
        actu1 = np.array([0, 0, 0, 1. * (0.5 + np.sin(0.005 * np.pi * i))])

        worm.set_actuation(actu1)
        scene.step()
    
    cam.stop_recording()
    if enable_vis:
        scene.viewer.stop()
    
if __name__ == "__main__":
    main()