import numpy as np
import genesis as gs
import time

import argparse
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    args = parser.parse_args()

    ########################## init ##########################
    gs.init(backend=gs.cpu)

    ########################## create a scene ##########################

    scene = gs.Scene(
    viewer_options = gs.options.ViewerOptions(
        camera_pos    = (0, -3.5, 2.5),
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
    
    # Add tabletop (cuboid)
    tabletop = scene.add_entity(
        gs.morphs.Box(
            size=(0.09, 0.07, 0.02),  # 90x70x20mm
            pos=(0.25, -0.2, 0.02),  # Initial position
        ),
        material=gs.materials.Rigid(
            friction=1,
        ),
        surface=gs.surfaces.Default(
            color=(0.5, 0.5, 0.5),
        )
    )
    
    # when loading an entity, you can specify its pose in the morph.
    franka = scene.add_entity(
        gs.morphs.MJCF(
            file='xml/franka_emika_panda/panda.xml',
            pos=(-0.25, -0.25, 0.0),    # Position: centered, half meter back
            euler=(0, 0, 0),         # Orientation in Euler angles (roll, pitch, yaw)
        )
    )

    # Adjust camera to match scene viewer
    cam = scene.add_camera(
        res=(1280, 720),
        pos=(0, -3.5, 2.5),  # Move camera back and up
        lookat=(0.0, 0.0, 0.5),  # Focus on center of structure
        fov=30,  # Slightly wider field of view
        GUI=False,
    )
    ########################## build ##########################
    scene.build()
    ############ Optional: set control gains ############
    # set positional gains (reduced for arm, original for gripper)
    motors_dof = np.arange(7)
    fingers_dof = np.arange(7, 9)
    franka.set_dofs_kp(
        np.array([2000, 2000, 1500, 1500, 1000, 1000, 1000, 100, 100]),  # Original gripper gains (100)
    )
    franka.set_dofs_kv(
        np.array([200, 200, 150, 150, 100, 100, 100, 10, 10]),  # Original gripper gains (10)
    )
    franka.set_dofs_force_range(
        np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
        np.array([ 87,  87,  87,  87,  12,  12,  12,  100,  100]),
    )
    import sys
    if sys.platform == "darwin":
        scene._visualizer._viewer._pyrender_viewer._renderer.dpscale = 1

    gs.tools.run_in_another_thread(fn=run_sim, args=(scene, args.vis, cam, franka, motors_dof, fingers_dof))
    if args.vis:
        scene.viewer.start()


def run_sim(scene, enable_vis, cam, franka, motors_dof, fingers_dof):
    # Create timestamp for unique video filename
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    video_filename = f'table_assembly_{timestamp}.mp4'
    
    # Start recording
    cam.start_recording()
    
    def step_and_render():
        scene.step()
        cam.render()  # Render camera view for recording
    
    # get the end-effector link
    end_effector = franka.get_link('hand')

    # Step 1: Pick and place the table leg
    # Move to pre-grasp pose above leg
    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([0.25, 0.1, 0.25]),  # Above leg position
        quat=np.array([0, 1, 0, 0]),
    )
    print("Moving to table leg...")
    qpos[-2:] = 0.04  # gripper open pos
    path = franka.plan_path(
        qpos_goal=qpos,
        num_waypoints=200,
    )
    for waypoint in path:
        franka.control_dofs_position(waypoint)
        step_and_render()
    
    # Reach down to leg
    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([0.25, 0.1, 0.13]),  # Lower to leg height
        quat=np.array([0, 1, 0, 0]),
    )
    franka.control_dofs_position(qpos[:-2], motors_dof)
    for i in range(100):
        step_and_render()

    # Grasp leg with stronger force
    franka.control_dofs_force(np.array([-2.0, -2.0]), fingers_dof)  # Increased from -0.5 to -2.0
    for i in range(100):
        step_and_render()

    # Lift leg
    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([0.25, 0.1, 0.25]),  # Lift above leg position
        quat=np.array([0, 1, 0, 0]),
    )
    franka.control_dofs_position(qpos[:-2], motors_dof)
    for i in range(200):
        step_and_render()

    # Move leg to assembly position
    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([0.0, 0.0, 0.25]),  # Center position
        quat=np.array([0, 1, 0, 0]),
    )
    franka.control_dofs_position(qpos[:-2], motors_dof)
    for i in range(200):
        step_and_render()

    # Place leg
    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([0.0, 0.0, 0.13]),  # Lower to ground
        quat=np.array([0, 1, 0, 0]),
    )
    franka.control_dofs_position(qpos[:-2], motors_dof)
    for i in range(100):
        step_and_render()

    # Release leg
    franka.control_dofs_position(np.array([0.08, 0.08]), fingers_dof)
    for i in range(100):
        step_and_render()

    # Step 2: Pick and place the tabletop
    # Move to pre-grasp pose above tabletop
    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([0.25, -0.2, 0.25]),  # Above tabletop position
        quat=np.array([0, 1, 0, 0]),
    )
    print("Moving to tabletop...")
    path = franka.plan_path(
        qpos_goal=qpos,
        num_waypoints=200,
    )
    for waypoint in path:
        franka.control_dofs_position(waypoint)
        step_and_render()

    # Reach down to tabletop
    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([0.25, -0.2, 0.13]),  # Lower to tabletop height
        quat=np.array([0, 1, 0, 0]),
    )
    franka.control_dofs_position(qpos[:-2], motors_dof)
    for i in range(100):
        step_and_render()

    # Grasp tabletop with stronger force
    franka.control_dofs_force(np.array([-2.0, -2.0]), fingers_dof)  # Increased from -0.5 to -2.0
    for i in range(100):
        step_and_render()

    # Lift tabletop
    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([0.25, -0.2, 0.25]),  # Lift above tabletop position
        quat=np.array([0, 1, 0, 0]),
    )
    franka.control_dofs_position(qpos[:-2], motors_dof)
    for i in range(200):
        step_and_render()

    # Move tabletop to assembly position
    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([0.0, 0.0, 0.25]),  # Center position
        quat=np.array([0, 1, 0, 0]),
    )
    franka.control_dofs_position(qpos[:-2], motors_dof)
    for i in range(200):
        step_and_render()

    # Place tabletop on leg
    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([0.0, 0.0, 0.13]),  # Height of leg (0.04) + half tabletop height (0.02)
        quat=np.array([0, 1, 0, 0]),
    )
    franka.control_dofs_position(qpos[:-2], motors_dof)
    for i in range(100):
        step_and_render()

    # Release tabletop
    franka.control_dofs_position(np.array([0.08, 0.08]), fingers_dof)
    for i in range(100):
        step_and_render()

    # Move arm to final position
    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([0.0, 0.0, 0.4]),  # Move up to safe position
        quat=np.array([0, 1, 0, 0]),
    )
    franka.control_dofs_position(qpos[:-2], motors_dof)
    for i in range(200):
        step_and_render()
    
    # Ensure all frames are captured before stopping
    for i in range(50):
        step_and_render()
    
    print(f"Saving video as: {video_filename}")
    cam.stop_recording(save_to_filename=video_filename, fps=60)
    
    time.sleep(1)
    
    if enable_vis:
        scene.viewer.stop()


if __name__ == "__main__":
    main()