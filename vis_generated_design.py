import pybullet as p
import pybullet_data

import time  # for delays
import numpy as np
import json
import argparse
import os
import sys
from PIL import Image

path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)  # this is a bit hacky... just in case the user doesnt have somo installed...
sys.path.insert(0, path)

from somo.sm_manipulator_definition import SMManipulatorDefinition
from somo.sm_actuator_definition import SMActuatorDefinition
from somo.sm_link_definition import SMLinkDefinition
from somo.sm_joint_definition import SMJointDefinition
from somo.sm_continuum_manipulator import SMContinuumManipulator
from somo.create_cmassembly_urdf import create_cmassembly_urdf

from somo.utils import load_constrained_urdf

from pybullet_utils.place_blocks_in_json import (
    move_until_contact,
    move_out_of_contact,
)

import sorotraj

def parse_arguments():
    """Parse command line arguments for configurable simulation"""
    parser = argparse.ArgumentParser(description='Unified Force Actuation Experiment')
    
    # Required arguments
    parser.add_argument('--json_file', '-j', required=True,
                       help='Path to JSON block file (e.g., bird.json, giraffe.json, octopus.json)')
    parser.add_argument('--trajectory_file', '-t', required=True,
                       help='Path to trajectory YAML file (e.g., trajectory_bird.yaml)')
    parser.add_argument('--joint_yaml', '-y', required=True,
                       help='Path to joint definition YAML file (e.g., bb_tail.yaml, bb_neck.yaml)')
    
    # Optional arguments
    parser.add_argument('--video', '-v', action='store_true', default=True,
                       help='Enable video recording (default: True)')
    parser.add_argument('--camera_distance', '-d', type=float, default=0.2,
                       help='Camera distance from object (default: 0.2)')
    parser.add_argument('--object_name', '-n', type=str, default=None,
                       help='Object name for output files (auto-detected if not provided)')
    parser.add_argument('--simulation_steps', '-s', type=int, default=60000,
                       help='Number of simulation steps (default: 60000)')
    parser.add_argument('--time_step', type=float, default=0.001,
                       help='Physics time step (default: 0.001)')
    
    return parser.parse_args()

def detect_object_dimensions(blocks_data):
    """Detect object dimensions from blocks data for camera targeting"""
    all_positions = []
    all_dimensions = []
    
    for block in blocks_data:
        pos = block["position"]
        all_positions.append([pos[0]/1000, pos[1]/1000, pos[2]/1000])  # Convert mm to m
        
        if block["shape"] == "cuboid":
            dims = [block["dimensions"]["x"]/1000, block["dimensions"]["y"]/1000, block["dimensions"]["z"]/1000]
        elif block["shape"] == "cylinder":
            if "radius" in block["dimensions"]:
                radius = block["dimensions"]["radius"]/1000
                height = block["dimensions"]["height"]/1000
            else:
                radius = block["dimensions"]["x"]/2000  # diameter/2
                height = block["dimensions"]["y"]/1000
            dims = [radius*2, radius*2, height]  # Approximate as box
        else:
            dims = [0.025, 0.025, 0.025]  # Default for joints
        
        all_dimensions.append(dims)
    
    if not all_positions:
        return 0.025, [0, 0, 0.025]  # Default values
    
    positions_array = np.array(all_positions)
    center = np.mean(positions_array, axis=0)
    
    # Calculate bounding box
    min_pos = np.min(positions_array, axis=0)
    max_pos = np.max(positions_array, axis=0)
    object_span = np.max(max_pos - min_pos)
    
    # Use maximum Z position as object height for camera targeting
    max_z = np.max(positions_array[:, 2])
    target_height = max(max_z, 0.025)
    
    return target_height, center.tolist()

def setup_camera(blocks_data, camera_distance=0.2):
    """Setup isometric camera view like pretty_visualize.py - AFTER blocks are placed"""
    # Simple fixed target like in individual files - objects will be near ground level after dropping
    target = [0, 0, 0.025]  # Fixed target near ground level like individual scripts
    
    # Camera setup for isometric view (same angles as pretty_visualize.py)
    cam_yaw = 45  # degrees - isometric angle
    cam_pitch = -35  # degrees - looking down from above
    
    # Calculate camera position from yaw, pitch, and distance
    yaw_rad = np.radians(cam_yaw)
    pitch_rad = np.radians(cam_pitch)
    
    # Camera position: isometric angle
    cam_x = target[0] + camera_distance * np.cos(pitch_rad) * np.cos(yaw_rad)
    cam_y = target[1] + camera_distance * np.cos(pitch_rad) * np.sin(yaw_rad)
    cam_z = target[2] + camera_distance * np.sin(-pitch_rad)  # Negative because pitch is negative
    
    eye = [cam_x, cam_y, cam_z]
    up = [0, 0, 1]
    
    print(f"Isometric camera settings (like pretty_visualize.py):")
    print(f"  Eye position: {eye}")
    print(f"  Target position: {target}")
    print(f"  Up vector: {up}")
    print(f"  Distance: {camera_distance:.2f}m, yaw: {cam_yaw:.1f}°, pitch: {cam_pitch:.1f}°")
    
    p.resetDebugVisualizerCamera(
        cameraDistance=camera_distance,
        cameraYaw=cam_yaw,
        cameraPitch=cam_pitch,
        cameraTargetPosition=target,
    )
    
    return target

def create_cuboid(block_data):
    # Convert dimensions from mm to meters
    dimensions = [
        block_data["dimensions"]["x"] / 1000,
        block_data["dimensions"]["y"] / 1000,
        block_data["dimensions"]["z"] / 1000
    ]
    
    # Convert position from mm to meters
    position = [
        block_data["position"][0] / 1000,
        block_data["position"][1] / 1000,
        block_data["position"][2] / 1000
    ]
    
    # Convert orientation from degrees to radians
    roll = np.radians(block_data["orientation"][0])
    pitch = np.radians(block_data["orientation"][1])
    yaw = np.radians(block_data["orientation"][2])
    
    # Create offset for SMLinkDefinition
    offset = [position[0], position[1], position[2], roll, pitch, yaw]
    
    # Calculate mass
    density = 1000  # kg/m^3
    mass = density * dimensions[0] * dimensions[1] * dimensions[2]
    
    # Create SMLinkDefinition
    link_def = SMLinkDefinition(
        shape_type="box",
        dimensions=dimensions,
        mass=mass,
        material_color=block_data["color"],
        inertial_values=[1, 0, 0, 1, 0, 1],
        material_name="base_color",
        origin_offset=offset,
    )
    
    # Load to PyBullet
    block_id = link_def.load_to_pybullet(physicsClient=physicsClient)
    
    if block_id is not None and block_id >= 0:
        # Set dynamics properties
        p.changeDynamics(block_id, -1, lateralFriction=0.8, restitution=0.1)
        return block_id
    else:
        print(f"Failed to create cuboid block {block_data['gpt_name']}")
        return None

def create_cylinder(block_data):
    # Convert dimensions from mm to meters
    if "radius" in block_data["dimensions"]:
        radius = block_data["dimensions"]["radius"] / 1000
        height = block_data["dimensions"]["height"] / 1000
    else:
        # Legacy format: x=diameter, y=height
        radius = block_data["dimensions"]["x"] / 2000  # diameter / 2
        height = block_data["dimensions"]["y"] / 1000
    
    # Convert position from mm to meters
    position = [
        block_data["position"][0] / 1000,
        block_data["position"][1] / 1000,
        block_data["position"][2] / 1000
    ]
    
    # Convert orientation from degrees to radians
    roll = np.radians(block_data["orientation"][0])
    pitch = np.radians(block_data["orientation"][1])
    yaw = np.radians(block_data["orientation"][2])
    
    # Create offset for SMLinkDefinition
    offset = [position[0], position[1], position[2], roll, pitch, yaw]
    
    # Calculate mass
    density = 1000  # kg/m^3
    mass = density * np.pi * radius**2 * height
    
    # Create SMLinkDefinition
    link_def = SMLinkDefinition(
        shape_type="cylinder",
        dimensions=[radius, height],
        mass=mass,
        material_color=block_data["color"],
        inertial_values=[1, 0, 0, 1, 0, 1],
        material_name="base_color",
        origin_offset=offset,
    )
    
    # Load to PyBullet
    block_id = link_def.load_to_pybullet(physicsClient=physicsClient)
    
    if block_id is not None and block_id >= 0:
        # Set dynamics properties
        p.changeDynamics(block_id, -1, lateralFriction=0.8, restitution=0.1)
        return block_id
    else:
        print(f"Failed to create cylinder block {block_data['gpt_name']}")
        return None

def create_joint(block_data, joint_yaml_path):
    # Convert position from mm to meters
    position = [
        block_data["position"][0] / 1000,
        block_data["position"][1] / 1000,
        block_data["position"][2] / 1000
    ]
    
    # Convert orientation from degrees to radians
    roll = np.radians(block_data["orientation"][0])
    pitch = np.radians(block_data["orientation"][1])
    yaw = np.radians(block_data["orientation"][2])
    
    # Load manipulator definition
    manipulator_definition = SMManipulatorDefinition.from_file(joint_yaml_path)
    
    # Create manipulator
    joint_block = SMContinuumManipulator(manipulator_definition)
    
    # Load to PyBullet
    startOr = p.getQuaternionFromEuler([roll, pitch, yaw])
    joint_block.load_to_pybullet(
        baseStartPos=position,
        baseStartOrn=startOr,
        baseConstraint="static",
        physicsClient=physicsClient,
    )
    
    return joint_block

def create_block(block_data, joint_yaml_path):
    if block_data["shape"] == "cuboid":
        return create_cuboid(block_data)
    elif block_data["shape"] == "cylinder":
        return create_cylinder(block_data)
    elif block_data["shape"] == "joint":
        return create_joint(block_data, joint_yaml_path)
    else:
        print(f"Shape {block_data['shape']} not supported")
        return None

def place_blocks(blocks_data, joint_yaml_path, drop=True):
    # Sort blocks by height (z-position) so lower blocks are placed first
    blocks_data_sorted = sorted(blocks_data, key=lambda x: x["position"][2])
    
    print(f"Blocks will be placed in this order (by height):")
    for i, block in enumerate(blocks_data_sorted):
        block_type = "🔗 Joint" if block["shape"] == "joint" else "📦 Regular"
        print(f"  {i+1}. {block['gpt_name']} ({block_type}) at height {block['position'][2]}mm")
    
    # Group joint blocks by base_block
    base_block_groups = {}
    block_ids = []
    
    print(f"\n=== PLACING BLOCKS SEQUENTIALLY ===")
    
    for i, block_data in enumerate(blocks_data_sorted):
        print(f"\nPlacing block {i+1}/{len(blocks_data_sorted)}: {block_data['gpt_name']}")
        
        if block_data["shape"] == "joint":
            # Handle joint blocks - group them, don't create directly
            base_block_name = block_data.get("base_block", "None")
            if base_block_name != "None":
                print(f"  🔗 Joint block - grouping with base block '{base_block_name}'")
                
                if base_block_name not in base_block_groups:
                    base_block_groups[base_block_name] = []
                
                base_block_groups[base_block_name].append(block_data)
                print(f"    ✅ Joint block '{block_data['gpt_name']}' grouped with base block '{base_block_name}'")
                continue
            else:
                print(f"  ⚠️ Joint block without base_block - skipping")
                continue
        else:
            # Regular blocks - create and drop them
            block_id = create_block(block_data, joint_yaml_path)
            if block_id is not None:
                block_ids.append(block_id)
                
                # Show actual dimensions in meters for debugging
                print(f"  Created block: {block_data['gpt_name']} at position {block_data['position']}")
                
                # Handle different dimension formats
                if block_data["shape"] == "cylinder":
                    if "radius" in block_data["dimensions"]:
                        radius = block_data["dimensions"]["radius"] / 1000
                        height = block_data["dimensions"]["height"] / 1000
                        print(f"  Dimensions: radius={radius:.3f}m, height={height:.3f}m")
                    else:
                        radius = block_data["dimensions"]["x"] / 2000
                        height = block_data["dimensions"]["y"] / 1000
                        print(f"  Dimensions: radius={radius:.3f}m, height={height:.3f}m")
                elif block_data["shape"] == "cuboid":
                    x_dim = block_data["dimensions"]["x"] / 1000
                    y_dim = block_data["dimensions"]["y"] / 1000
                    z_dim = block_data["dimensions"]["z"] / 1000
                    print(f"  Dimensions: {x_dim:.3f}m x {y_dim:.3f}m x {z_dim:.3f}m")
                
                # Get initial position
                pos, ori = p.getBasePositionAndOrientation(block_id)
                print(f"  📍 Initial position: {pos}")
                
                if drop:
                    # Drop the block using move_until_contact and move_out_of_contact
                    print(f"  📦 Regular block - dropping...")
                    move_until_contact(block_id, direction=[0, 0, -1], step_size=0.001, debug=False)
                    move_out_of_contact(block_id)
                    
                    # Get final position after drop
                    pos_after, ori_after = p.getBasePositionAndOrientation(block_id)
                    print(f"  📍 Final position after drop: {pos_after}")
                    
                    # Wait a bit for physics to settle
                    for _ in range(100):
                        p.stepSimulation()
                        time.sleep(0.001)  # 1ms delay for physics settling
                    
                    # Don't make central body static for octopus locomotion
                    if block_data['gpt_name'] not in ['central_body']:
                        p.changeDynamics(block_id, -1, mass=0)  # Set mass to 0 to make it static
                        print(f"  🔒 Made block '{block_data['gpt_name']}' static after drop (mass=0)")
                    else:
                        print(f"  🐙 Kept '{block_data['gpt_name']}' dynamic for locomotion")
                
            else:
                print(f"  ❌ Failed to create block: {block_data['gpt_name']}")
    
    return block_ids, base_block_groups

def create_joint_assemblies(block_ids, base_block_groups, blocks_data, joint_yaml_path):
    """Create joint manipulators directly"""
    print(f"\n=== CREATING JOINT MANIPULATORS ===")
    
    # Initialize joint manipulators list
    if not hasattr(p, 'joint_manipulators'):
        p.joint_manipulators = []
    
    for base_name, joint_blocks in base_block_groups.items():
        if not joint_blocks:
            continue
            
        print(f"Creating joint manipulators for base block '{base_name}' with {len(joint_blocks)} joint(s)")
        
        for joint_block in joint_blocks:
            print(f"  - Creating manipulator for joint: {joint_block['gpt_name']}")
            
            # Create joint manipulator directly
            joint_manipulator = create_joint(joint_block, joint_yaml_path)
            if joint_manipulator is not None:
                # Store both manipulator and block data
                manipulator_info = {
                    'manipulator': joint_manipulator,
                    'block_data': joint_block
                }
                p.joint_manipulators.append(manipulator_info)
                print(f"    ✅ Joint manipulator created for '{joint_block['gpt_name']}'")
            else:
                print(f"    ❌ Failed to create joint manipulator for '{joint_block['gpt_name']}'")
    
    print(f"\n=== JOINT MANIPULATORS SUMMARY ===")
    if p.joint_manipulators:
        print(f"Created {len(p.joint_manipulators)} joint manipulators:")
        for i, manipulator_info in enumerate(p.joint_manipulators):
            manipulator = manipulator_info['manipulator']
            block_data = manipulator_info['block_data']
            print(f"  {i+1}. {block_data['gpt_name']} at position {block_data['position']}")
    else:
        print("No joint manipulators created")
    
    return block_ids

def main():
    global physicsClient
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Auto-detect object name if not provided
    if args.object_name is None:
        args.object_name = os.path.splitext(os.path.basename(args.json_file))[0]
    
    print(f"🚀 Starting Unified Force Actuation Experiment")
    print(f"📁 JSON file: {args.json_file}")
    print(f"🎯 Trajectory file: {args.trajectory_file}")
    print(f"🔗 Joint YAML: {args.joint_yaml}")
    print(f"🎬 Video recording: {args.video}")
    print(f"📷 Camera distance: {args.camera_distance}m")
    print(f"🏷️ Object name: {args.object_name}")
    
    ######## SIMULATION SETUP ########
    
    ### prepare everything for the physics client / rendering
    ## Pretty rendering
    opt_str = "--background_color_red=1.0 --background_color_green=1.0 --background_color_blue=1.0"  # this opens the gui with a white background and no ground grid
    cam_width, cam_height = 1920, 1640
    if cam_width is not None and cam_height is not None:
        opt_str += " --width=%d --height=%d" % (cam_width, cam_height)
    
    physicsClient = p.connect(
        p.GUI, options=opt_str
    )  # starts the physics client with the options specified above. replace p.GUI with p.DIRECT to avoid gui
    
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    
    # Load blocks from JSON file
    with open(args.json_file, 'r') as f:
        blocks_data = json.load(f)
    print(f"Successfully loaded {len(blocks_data)} blocks from JSON file")
    
    ## Set physics parameters and simulation properties
    p.setGravity(0, 0, -10)
    p.setPhysicsEngineParameter(enableConeFriction=1)
    p.setRealTimeSimulation(0)  # this is necessary to enable torque control
    
    ## Specify time steps
    p.setTimeStep(args.time_step)
    
    ### load all the objects into the environment
    # Create a simple ground plane
    plane_collision_shape = p.createCollisionShape(p.GEOM_PLANE)
    plane_visual_shape = p.createVisualShape(p.GEOM_PLANE, rgbaColor=[0.8, 0.8, 0.8, 1.0])
    planeId = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=plane_collision_shape,
        baseVisualShapeIndex=plane_visual_shape,
        basePosition=[0, 0, 0]
    )
    p.changeDynamics(planeId, -1, lateralFriction=1)  # set ground plane friction
    
    # Sort blocks by height (z-position) so lower blocks are placed first
    blocks_data_sorted = sorted(blocks_data, key=lambda x: x["position"][2])
    print(f"Blocks will be placed in this order (by height):")
    for i, block in enumerate(blocks_data_sorted):
        block_type = "🔗 Joint" if block["shape"] == "joint" else "📦 Regular"
        print(f"  {i+1}. {block['gpt_name']} ({block_type}) at height {block['position'][2]}mm")
    
    # Create blocks in PyBullet sequentially with dropping
    block_ids, base_block_groups = place_blocks(blocks_data_sorted, args.joint_yaml)
    
    # Create joint assemblies
    block_ids = create_joint_assemblies(block_ids, base_block_groups, blocks_data_sorted, args.joint_yaml)
    
    print(f"\n=== BLOCK PLACEMENT COMPLETE ===")
    print(f"Successfully created and placed {len(block_ids)} total blocks in PyBullet")
    print(f"  - Regular blocks: {len([b for b in blocks_data if b['shape'] != 'joint'])}")
    print(f"  - Joint blocks: {len([b for b in blocks_data if b['shape'] == 'joint'])}")
    print(f"  - Total blocks: {len(block_ids)}")
    
    # Setup camera AFTER blocks are placed and settled
    setup_camera(blocks_data, args.camera_distance)
    
    # Apply contact properties
    contact_properties = {
        "lateralFriction": 1,
        "restitution": 0.5,
    }
    
    if hasattr(p, 'joint_manipulators'):
        for manipulator_info in p.joint_manipulators:
            manipulator = manipulator_info['manipulator']
            block_data = manipulator_info['block_data']
            print(f"Applying contact properties to manipulator: {block_data['gpt_name']}")
            
            # Apply to the manipulator base
            p.changeDynamics(
                manipulator.bodyUniqueId, 
                -1, 
                lateralFriction=contact_properties["lateralFriction"],
                restitution=contact_properties["restitution"]
            )
    
    # ######## PRESCRIBE A TRAJECTORY ########
    print(f"\n=== LOADING TRAJECTORY ===")
    print(f"Trajectory file: {args.trajectory_file}")
    
    try:
        traj = sorotraj.TrajBuilder(graph=False)
        traj.load_traj_def(args.trajectory_file)
        trajectory = traj.get_trajectory()
        print(f"✅ Trajectory loaded successfully")
        print(f"Trajectory Loaded: {args.trajectory_file}")
        
        interp = sorotraj.Interpolator(trajectory)
        actuation_fn = interp.get_interp_function(
            num_reps=1, speed_factor=1.2, invert_direction=False, as_list=False
        )
        print(f"✅ Trajectory interpolator created successfully")
    except Exception as e:
        print(f"❌ Error loading trajectory: {e}")
        print(f"🔧 Running simulation without trajectory (physics only)")
        actuation_fn = None
    
    ######## EXECUTE SIMULATION ########
    # if desired, start video logging - this goes before the run loop
    if args.video:
        vid_filename = os.path.join(os.path.dirname(__file__), f"{args.object_name}_simulation.mp4")
        print(f"🎬 Starting video recording to: {vid_filename}")
        logIDvideo = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, vid_filename)
        print(f"📹 Video logging ID: {logIDvideo}")
    
    # this for loop is the actual simulation
    if hasattr(p, 'joint_manipulators') and len(p.joint_manipulators) > 0:
        print(f"\n🎭 Starting force actuation simulation with {len(p.joint_manipulators)} joint manipulator(s)")
        
        # Get the first joint manipulator (works for single joint objects like bird/giraffe)
        manipulator_obj = p.joint_manipulators[0]['manipulator']
        joint_name = p.joint_manipulators[0]['block_data']['gpt_name']
        
        # Determine actuator layout dynamically
        num_actuators = len(manipulator_obj.manipulator_definition.actuator_definitions)
        print(f"Joint manipulator: {joint_name}")
        print(f"Number of actuators: {num_actuators}")
        
        # For 2 actuators with 2 axes each: [0,0,1,1] and [0,1,0,1]
        actuator_nrs = []
        axis_nrs = []
        for act in range(num_actuators):
            actuator_nrs.extend([act, act])  # Each actuator has 2 axes
            axis_nrs.extend([0, 1])         # Axis 0 and 1 for each actuator
        
        print(f"Actuator numbers: {actuator_nrs}")
        print(f"Axis numbers: {axis_nrs}")
        
        # Check if we have a valid trajectory
        if actuation_fn is None:
            print(f"❌ No valid trajectory - running physics simulation only")
            for i in range(args.simulation_steps):
                p.stepSimulation()
                if i % 10000 == 0:
                    progress = (i / args.simulation_steps) * 100
                    print(f"📦 Physics simulation progress: {progress:.1f}%")
        else:
            print(f"✅ Starting trajectory-based actuation")
            for i in range(args.simulation_steps):
                try:
                    torques = actuation_fn(i * args.time_step)
                    
                    # Debug torques on first few steps
                    if i < 5:
                        print(f"Step {i}: time={i * args.time_step:.3f}s, torques={torques}")
                    
                    # applying the control torques
                    manipulator_obj.apply_actuation_torques(
                        actuator_nrs=actuator_nrs,
                        axis_nrs=axis_nrs,
                        actuation_torques=torques.tolist(),
                    )
                except Exception as e:
                    if i % 1000 == 0:  # Print occasionally to avoid spam
                        print(f"Warning: Could not apply torques at step {i}: {e}")
                
                p.stepSimulation()
                
                # Print progress occasionally
                if i % 10000 == 0:
                    progress = (i / args.simulation_steps) * 100
                    print(f"🎭 Simulation progress: {progress:.1f}%")
    else:
        print(f"\n📦 No joint manipulators found - running physics simulation only")
        for i in range(args.simulation_steps):
            p.stepSimulation()
            
            # Print progress occasionally
            if i % 10000 == 0:
                progress = (i / args.simulation_steps) * 100
                print(f"📦 Physics simulation progress: {progress:.1f}%")
    
    ######## CLEANUP AFTER SIMULATION ########
    # this goes after the run loop
    if args.video:
        print("🛑 Stopping video recording...")
        p.stopStateLogging(logIDvideo)
        print(f"✅ Video saved to: {vid_filename}")
        if os.path.exists(vid_filename):
            print(f"📁 Video file size: {os.path.getsize(vid_filename)} bytes")
        else:
            print("❌ Video file was not created!")
    
    # ... aaand disconnect pybullet
    p.disconnect()
    print(f"🏁 Simulation complete for {args.object_name}")

if __name__ == "__main__":
    main()
