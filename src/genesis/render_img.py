import numpy as np
import genesis as gs
import time
from scipy.spatial.transform import Rotation
import cv2
import os
from datetime import datetime
from PIL import Image
import argparse

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    args = parser.parse_args()

    ########################## init ##########################
    gs.init(backend=gs.cpu)

    ########################## create a scene ##########################

    scene = gs.Scene(
    viewer_options=gs.options.ViewerOptions(
        camera_fov=30,
    ),
    sim_options=gs.options.SimOptions(
        dt       = 4e-3,
        substeps = 10,
    ),
    rigid_options=gs.options.RigidOptions(
        gravity=(0.0, 0.0, -10.0),
    ),
    mpm_options=gs.options.MPMOptions(
        lower_bound   = (-0.5, -1.0, 0.0),
        upper_bound   = (0.5, 1.0, 1),
    ),
    vis_options=gs.options.VisOptions(
        visualize_mpm_boundary = True,
    ),
    show_viewer = args.vis,
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

    obj_elastic = scene.add_entity(
    material=gs.materials.MPM.Elastic(),
    morph=gs.morphs.Box(
        pos  = (0.0, -0.5, 0.25),
        size = (0.2, 0.2, 0.2),
    ),
    surface=gs.surfaces.Default(
        color    = (1.0, 0.4, 0.4),
        vis_mode = 'visual',
    ),
)

    obj_sand = scene.add_entity(
        material=gs.materials.MPM.Liquid(),
        morph=gs.morphs.Box(
            pos  = (0.0, 0.0, 0.25),
            size = (0.3, 0.3, 0.3),
        ),
        surface=gs.surfaces.Default(
            color    = (0.3, 0.3, 1.0),
            vis_mode = 'particle',
        ),
    )

    obj_plastic = scene.add_entity(
        material=gs.materials.MPM.ElastoPlastic(),
        morph=gs.morphs.Sphere(
            pos  = (0.0, 0.5, 0.35),
            radius = 0.1,
        ),
        surface=gs.surfaces.Default(
            color    = (0.4, 1.0, 0.4),
            vis_mode = 'particle',
        ),
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
    scene.step()

    import sys
    if sys.platform == "darwin":
        scene._visualizer._viewer._pyrender_viewer._renderer.dpscale = 2

    gs.tools.run_in_another_thread(fn=run_sim, args=(scene, args.vis, cam, obj_elastic, obj_sand, obj_plastic))
    if args.vis:
        scene.viewer.start()


def run_sim(scene, enable_vis, cam, obj_elastic, obj_sand, obj_plastic):
    horizon = 1000
    for i in range(horizon):
        scene.step()
    if enable_vis:
        scene.viewer.stop()
    
if __name__ == "__main__":
    main()