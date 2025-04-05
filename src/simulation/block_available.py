import genesis as gs
import datetime
import cv2
import time
import json
import os
import argparse
from concurrent.futures import ThreadPoolExecutor


def main():
    parser = argparse.ArgumentParser(description='Create blocks in Genesis from a state JSON file')
    parser.add_argument("-v", "--vis", action="store_true", default=False, help="Enable visualization")
    parser.add_argument("-f", "--file", type=str, required=True, help="Path to the state JSON file")
    parser.add_argument("-e", "--evaluate", action="store_true", default=False, help="Evaluate semantic recognizability")
    args = parser.parse_args()

    ########################## init ##########################
    gs.init(backend=gs.cpu)

    # Load state file to get object name
    state = load_simulated_blocks(args.file)

    ########################## create a scene ##########################
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.3, 0.3, 0.3),  # Much closer view
            camera_lookat=(0.0, 0.0, 0.1),  # Looking slightly up from ground
            camera_fov=40,
        ),
        show_viewer=args.vis,
        rigid_options=gs.options.RigidOptions(
            dt=0.01,
            gravity=(0.0, 0.0, -10.0),
        ),
    )

    ########################## entities ##########################
    # Add ground plane
    plane = scene.add_entity(gs.morphs.Plane())
    
    # Load block positions and sizes from state file

    # Add entities to the scene using the block dictionary
    for name, data in state.items():
        if data["shape"] == "cylinder":
            # For cylinders, use Cylinder morph with radius and height
            scene.add_entity(
                morph=gs.morphs.Cylinder(
                    pos=data["pos"],
                    radius=data["size"][0],  # First element is radius
                    height=data["size"][1]   # Second element is height
                )
            )
        else:
            # For boxes, use Box morph with size (length, width, height)
            scene.add_entity(
                morph=gs.morphs.Box(
                    pos=data["pos"],
                    size=data["size"]
                )
            )

    # Add camera for recording
    cam = scene.add_camera(
        res=(1280, 720),
        pos=(0.5, 0.5, 0.4),  # Much closer recording position
        lookat=(0.0, 0.0, 0.1),  # Looking slightly up from ground
        fov=40,
        GUI=False,
    )

    ########################## build ##########################
    scene.build()

    import sys
    if sys.platform == "darwin":
        scene._visualizer._viewer._pyrender_viewer._renderer.dpscale = 1

    # Handle simulation and viewer based on platform
    if args.vis and sys.platform == "darwin":
        # On macOS, run simulation in a separate thread
        with ThreadPoolExecutor() as executor:
            # Start simulation in a separate thread
            future = executor.submit(run_sim, scene, args.vis, cam, object_name)
            
            # Start viewer in the main thread
            scene.viewer.start()
            
            # Wait for simulation to complete
            paths = future.result()
    else:
        # On other platforms or without visualization, run synchronously
        paths = run_sim(scene, args.vis, cam, object_name)

    # Evaluate semantic recognizability if requested
    if args.evaluate and paths:
        try:
            
            print("\nEvaluating semantic recognizability...")
            evaluator = SemanticRecognizability()
            result = evaluator.evaluate_single(paths["image_path"], state["object_name"])
            
            print("\nSemantic Recognizability Results:")
            print(f"Top-1 Accuracy: {100 if result['top_1_accuracy'] else 0}%")
            print(f"Average Rank: {result['average_rank']}")
            print(f"Relative Rank: {result['relative_rank']:.1f}%")
            print("\nFull Analysis:")
            print(result['full_response'])
            
        except Exception as e:
            print(f"\nError evaluating semantic recognizability: {str(e)}")
            print(f"Paths returned from simulation: {paths}")  # Debug info

def extract_last_frame(video_path: str, output_path: str) -> None:
    """
    Extract the last frame from a video file and save it as an image.
    
    Args:
        video_path (str): Path to the video file
        output_path (str): Path where to save the extracted frame
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    # Get total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Set frame position to last frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
    
    # Read the last frame
    ret, frame = cap.read()
    
    if ret:
        # Save the frame as image
        cv2.imwrite(output_path, frame)
        print(f"Last frame saved as {output_path}")
    else:
        print("Error: Could not read the last frame")
    
    # Release video capture object
    cap.release()

def run_sim(scene, enable_vis, cam, object_name):
    i = 0
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    video_filename = f'videos/{object_name}_{timestamp}.mp4'
    image_filename = f'images/{object_name}_{timestamp}.jpg'
    
    # Create necessary directories if they don't exist
    os.makedirs('videos', exist_ok=True)
    os.makedirs('images', exist_ok=True)
    
    # Run simulation
    cam.start_recording()
    for i in range(120):
        scene.step()
        cam.render()
    
    # Stop recording and save video
    cam.stop_recording(save_to_filename=video_filename, fps=60)
    print(f"Video saved as {video_filename}")
    
    # Wait a moment to ensure video is fully written
    time.sleep(1)
    
    # Extract and save the last frame from the video
    extract_last_frame(video_filename, image_filename)

    if enable_vis:
        scene.viewer.stop()
    
    # Return paths for potential further processing
    return {
        "video_path": video_filename,
        "image_path": image_filename,
        "timestamp": timestamp
    }

def load_simulated_blocks(json_path: str) -> dict:
    """
    Load and convert simulated blocks from JSON file to a dictionary.
    
    Args:
        json_path (str): Path to the simulated blocks JSON file
        
    Returns:
        dict: Dictionary containing block information with the following structure:
            {
                "block_name": {
                    "dimensions": {"x": float, "y": float, "z": float} or {"radius": float, "height": float},
                    "shape": str,
                    "number_available": int
                }
            }
    """
    try:
        with open(json_path, 'r') as f:
            blocks_data = json.load(f)
        return blocks_data
    except FileNotFoundError:
        print(f"Error: Could not find file {json_path}")
        return {}
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in file {json_path}")
        return {}

if __name__ == "__main__":
    data = load_simulated_blocks("/Users/pqnhhh/Documents/GitHub/multi-agent-block-desgin/data/simulated_blocks.json")
    for name, data in data.items():
        print(name, data)