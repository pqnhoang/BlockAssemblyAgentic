import os
import base64
import torch
import torchvision
from openai import OpenAI
from PIL import Image
import numpy as np
import requests
import json
import sys
import pybullet as p
import pybullet_data
import time
from dotenv import load_dotenv
from typing import Dict, List
from pathlib import Path
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_PATH, "gpt_caching")
POSITIONS_DIR = os.path.join(BASE_PATH, "final_results/positions")
sys.path.append(BASE_PATH)

#DESIGN CLASS
from src.structure.block import Block
from src.structure.structure import Structure
from src.structure.assembly import Assembly
from src.pybullet.pybullet_axes import get_imgs
from src.utils.utils import get_last_json_as_dict, load_from_json, save_to_json, markdown_json, slugify
from src.prompt.prompt import prompt_with_caching
import shutil
from somo.sm_manipulator_definition import SMManipulatorDefinition
from somo.sm_link_definition import SMLinkDefinition
from somo.create_cmassembly_urdf import create_cmassembly_urdf

query_image_path = os.path.join(BASE_PATH, "imgs/block.png")
device = "cuda:0" if torch.cuda.is_available() else "cpu"
# Load environment variables from .env file
load_dotenv()

## OpenAI API
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file or environment variables.")
OPENAI_CLIENT = OpenAI(api_key=api_key)
OPENAI_CLIENT = OpenAI(api_key=api_key[0])
EMBEDDING_MODEL = "text-embedding-ada-002"
TEMPERATURE = 0.5



def blocks_from_json(json_data):
    blocks = []
    for block_data in json_data:
        if block_data["shape"] == "cuboid":
            dimensions = [
                block_data["dimensions"]["x"],
                block_data["dimensions"]["y"],
                block_data["dimensions"]["z"],
            ]
        elif block_data["shape"] == "cylinder" or block_data["shape"] == "cone":
            dimensions = [
                block_data["dimensions"]["radius"],
                block_data["dimensions"]["height"],
            ]
        elif block_data["shape"] == "joint":
            dimensions = [
                block_data["dimensions"]["x"],
                block_data["dimensions"]["y"],
                block_data["dimensions"]["z"],
            ]
        else:
            raise ValueError(f"Invalid shape {block_data['shape']}")

        block = Block(
            id=999,  # id gets updated by place blocks call, otherwise it's unknown
            gpt_name=block_data["name"],
            block_name="",
            shape=block_data["shape"],
            dimensions=dimensions,
            position=[
                block_data["position"]["x"],
                block_data["position"]["y"],
                1 * 1000,
            ],
            orientation=p.getQuaternionFromEuler([0, 0, np.radians(block_data["yaw"])]),
            color=block_data["color"],
        )
        blocks.append(block)

    return blocks

def process_available_blocks(blocks):
    available_blocks = []
    for block_name, block in blocks.items():
        block_shape = block["shape"]
        block_dimensions = block["dimensions"]
        number_available = block["number_available"]
        available_blocks.append(
            {
                "shape": block_shape,
                "dimensions": block_dimensions,
                "number_available": number_available,
            }
        )
    return available_blocks

def normalize_position(position) -> np.ndarray:
    """Chuẩn hóa vị trí về dạng mảng NumPy [x, y, z]."""
    if isinstance(position, dict):
        return np.array([position.get('x', 0), position.get('y', 0), position.get('z', 0)], dtype=float)
    if isinstance(position, list) and len(position) == 2:
        return np.array([position[0], position[1], 0], dtype=float)
    return np.array(position[:3], dtype=float)

class IsometricImageJoint:
    """
    A class to represent a isometric image of a object.
    """
    def __init__(self, object_name: str, positions: Dict | Path = None, structure_img: Image.Image | Path = None, available_blocks_path: str = None):
        """
        Initialize a IsometricImageJoint class object by providing the query and feed back image. Then we create a scene object to perform build-in functions.
        Parameters
        ----------
        object_name : str
            The name of the object to be built.
        positions : json
            An json contains postions of building blocks. If None, the positions will be set to None.
        available_blocks_path : str, optional
            Path to the JSON file containing available blocks. If None, uses default path.
            
        """

        self.object_name = slugify(object_name)
        self.structure_dir = os.path.join(SAVE_DIR, self.object_name)
        # Load available blocks
        if available_blocks_path is None:
            available_blocks_path = os.path.join(BASE_PATH, "data/simulated_blocks_joint.json")
        available_blocks = load_from_json(available_blocks_path)
        self.available_blocks = process_available_blocks(available_blocks)

        # Create the structure directory if it doesn't exist
        if isinstance(positions, str):
            try:
                if os.path.exists(positions) and os.path.getsize(positions) > 0:
                    with open(positions, 'r') as positions_data:
                        self.positions = json.load(positions_data)
                else:
                    self.positions = None
            except json.JSONDecodeError:
                print(f"Warning: File '{positions}' invalid JSON format or empty. Setting positions to None. Please check the file content.")
                self.positions = None
        elif isinstance(positions, dict):
            self.positions = positions
        else:
            self.positions = None
        
        # Load positions from JSON file if provided, otherwise set to None
        if self.positions is None: 
            self.blocks = []
        else:
            self.blocks = blocks_from_json(self.positions)
        
        # Load structure image if provided, otherwise set to None
        if isinstance(structure_img, Image.Image):
            self.structure_image = structure_img
        elif isinstance(structure_img, Path) or isinstance(structure_img, str):
            if os.path.exists(structure_img):
                self.structure_image = Image.open(structure_img)
            else:
                self.structure_image = None
        else:
            self.structure_image = None
        self.structure = None
        # # Initialize variables
        self.main_llm_context: List[Dict] = []
        self.eval_llm_context: List[Dict] = []
    
    def llm_query(self, question, context=None, long_answer=True, queues=None):
        """Answers a text question using GPT-4o-mini. The input question is always a formatted string with a variable in it.

        Parameters
        ----------
        query: str
            the text question to ask. Must not contain any reference to 'the image' or 'the photo', etc.
        """
        # Use the first image (top view) for the query
        self.original_img[1].save(query_image_path,"PNG")
        with open(query_image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')

        response = OPENAI_CLIENT.chat.completions.create(
          model="gpt-4o",
          messages=[
            {
              "role": "user",
              "content": [
                {
                  "type": "text",
                  "text": question,
                },
                {
                  "type": "image_url",
                  "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    "detail": "high"
                  }
                }
              ]
            }
          ],
          max_tokens=300,
        )

        response_message = response.choices[0].message.content
        return response_message
    

    def describe_object(self, iter=0):
        """
        Describe the object in the image.
        """

        prompt = f"""
        I'm working on constructing a block tower that represents a(n) {self.object_name}. I need a concise, qualitative description of the design that captures its essence in a minimalistic style. The design should focus on simplicity, avoiding unnecessary complexity while still conveying the key features of a(n) {self.object_name}. The description should highlight the overall structure and proportions, emphasizing how the block arrangement reflects the object's shape and form. However the design shouldn't be too large, too wide, or too tall.
        This design can incorporate both rigid blocks (cuboids, cylinders) and flexible joint blocks that allow for articulated connections and curved forms. For objects requiring flexibility, bending, or articulation (like animals, human figures, or curved structures), joint blocks should be strategically used to create smooth transitions and natural poses. The joint blocks enable the structure to have flexible segments that can bend and move, making it more lifelike and dynamic.
        """.strip()
        self.main_llm_context = []

        response, updated_context = prompt_with_caching(
            prompt,
            self.main_llm_context,
            self.structure_dir,
            "description",
            cache=True,
            temeprature=TEMPERATURE,
            i=iter,
    )
        self.main_llm_context = updated_context
        return response
    
    def make_plan(self, description, iter=0):
        """
        Make a plan to assemble the object.
        """

        prompt = f"""
        Here's a description of the layout of a {self.object_name}:
        {description}

        You have the following blocks available: 
        {markdown_json(self.available_blocks)}
        Write a plan for how to assemble a {self.object_name} using the available blocks. Use blocks as needed while respecting the number available constraint. 

        Explain which blocks to use and their shape and dimensions. 

        Explain the overall orientation of the structure.

        Explain each block's role in the structure. 

        Explain how the blocks should stack on top of each other (they can also just be placed on the ground). 

        Do not overcomplicate the design. Try to use a minimal number of blocks to represent the key components of a {self.object_name}. Avoid making structures that are too tall, wide, or complex.

        Only consider the main key components of a {self.object_name}, not minor details that are hard to represent with blocks. 
        Use a minimal amount of blocks and keep it simple, just enough so that it looks like a {self.object_name}.

        The dimensions of a cuboid are given as x, y, and z, which define the size of the block. You can rearrange these dimensions to fit your design requirements. For instance, if you need to place a block "vertically" with its longest side along the z-axis, but the dimensions are listed as x: 30, y: 30, z: 10, you can adjust them to x: 10, y: 30, z: 30 to achieve the desired orientation. Ensure the x and y dimensions are also consistent with the rest of the design.

        Cylinders are always positioned "upright," with the flat sides parallel to the ground and their radius extending along the x and y axes.

        Joint blocks are flexible blocks composed of 8 interconnected segments that work as an articulated chain. They represent soft, bendable parts of objects requiring natural curves and movement - ideal for flexible limbs (arms, legs, tentacles, tails), curved body parts (necks, spines), and organic connections between rigid components. Joint blocks can bend and articulate to create smooth curves and lifelike poses that rigid blocks cannot achieve, enhancing the overall authenticity and expressiveness of the {self.object_name}.

        Decide a semantic name for the block for the role it represents in the structure. 
        Decide the colors of each block to look like a {self.object_name}. Color is an rgba array with values from 0 to 1.
        """
        
        response,updated_context = prompt_with_caching(
            prompt,
            self.main_llm_context,
            self.structure_dir,
            "main_plan",
            cache=True,
            temeprature=TEMPERATURE,
            i=iter,
    )
        self.main_llm_context = updated_context
        return response
    def order_blocks(self, plan, iter=0):
        """
        Order the blocks in the plan.
        """
        
        prompt = f"""
        Given the blocks described in the plan {plan}, I will place and stack these blocks one at a time by lowering them from a very tall height.

        Please describe the sequence in which the blocks should be placed to correctly form a {self.object_name} structure. This means that blocks at the bottom should be placed first, followed by the higher blocks, so that the upper blocks can be stacked on top of the lower ones. Also note that it is difficult to stack blocks on top of a cone, so avoid placing blocks directly on top of cones.

        For each block, specify whether it will be placed on the ground or on top of another block. If a block will be supported by multiple blocks, mention all of them. Ensure that the blocks are placed in a way that they remain stable and won't topple over when the physics simulation is run. Blocks cannot hover without support.
        """.strip()

        
        response, updated_context = prompt_with_caching(
            prompt,
            self.main_llm_context,
            self.structure_dir,
            "order_plan",
            cache=True,
            temeprature=TEMPERATURE,
            i=iter,
    )
        self.main_llm_context = updated_context
        return response
    def decide_position(self, order, iter=0):
        prompt = f"""
        With the stacking order determined {order}, I now need to know the x and y positions, as well as the yaw angle (in degrees), for each block to build a {self.object_name} structure.

        The x and y coordinates should represent the center of each block. The yaw angle refers to the rotation around the z-axis in degrees. Remember, you can swap the dimensions of blocks to adjust their configuration.

        Ensure that blocks at similar heights in the structure are spaced out in x and y so that they don't collide.

        Make sure the structure is roughly centered at the origin (0, 0), and that each block stacks correctly on the specified blocks (or the ground). Every block must have a stable base to prevent it from falling. 

        Consider the dimensions of the blocks when determining the x, y positions. Provide your reasoning for the chosen x and y positions and the yaw angle for each block.

        Output a JSON following this format:
        {markdown_json(
            [
                {
                    "name": "support1",
                    "shape": "cylinder",
                    "dimensions": {"radius": 20, "height": 40},
                    "color": [0.5, 0.5, 0.5, 1],
                    "position": {"x": -50, "y": 0},
                    "yaw": 0,
                },
                {
                    "name": "support2",
                    "shape": "cylinder",
                    "dimensions": {"radius": 20, "height": 40},
                    "color": [0.5, 0.5, 0.5, 1],
                    "position": {"x": 50, "y": 0},
                    "yaw": 0,
                },
                {
                    "name": "deck",
                    "shape": "cuboid",
                    "dimensions": {"x": 100, "y": 50, "z": 20},
                    "color": [0.5, 0.5, 0.5, 1],
                    "position": {"x": 0, "y": 0},
                    "yaw": 45,
                },
            ]
        )}
        """
        response, updated_context = prompt_with_caching(
            prompt,
            self.main_llm_context,
            self.structure_dir,
            "decide_position",
            cache=True,
            temeprature=TEMPERATURE,
            i=iter,
        )
        self.main_llm_context = updated_context
        json_output = get_last_json_as_dict(response)
        blocks = blocks_from_json(json_output)
        self.blocks = blocks
        self.positions = json_output
        return json_output
    
    def get_stability_correction(self, to_build, unstable_block: Block, pos_delta, structure_json, x_img, y_img, iter=0):
        prompt = [
            f"""
        {markdown_json(structure_json)}

        While building the {to_build} by placing blocks one at a time in the order you specified by the JSON above, I noticed that block {unstable_block.gpt_name} is unstable and falls. 
        The block moved by {pos_delta[0]:.2f} mm in the x direction and {pos_delta[1]:.2f} mm in the y direction.
        Please adjust the position of block {unstable_block.gpt_name} (And potentially other blocks) to make the structure more stable.
        Make sure every block has a stable base to rest on.

        Output the JSON with your corrections following the same format and provide some reasoning for the changes you made. Feel free to correct other parts of the structure if they appear incorrect or to add, change, or remove blocks.

        Here is an orthographic image of the side view of the structure with the y-axis pointing to the right and the z-axis pointing up. {unstable_block.gpt_name} is highlighted in red while the other blocks are colored in white.
        """,
                x_img,
            f"""
        Here is an orthographic image of the side view of the structure with the x-axis pointing to the right and the z-axis pointing up. {unstable_block.gpt_name} is highlighted in red while the other blocks are colored in white.
        """,
        y_img,
            f"""
        Describe what you see in these images and use them to help inform your correction. Then, provide the ouptut JSON in the proper format.
        """
        ]
        response, stability_context = prompt_with_caching(
            prompt, [], self.structure_dir, f"stability_correction_{iter}", cache=True, i=iter
        )
        return response, stability_context
    
    def make_structure(self, positions: Dict):
        """
        Make a structure from a dictionary of positions.
        """
        if not p.isConnected():
            p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        structure_dir = os.path.join(BASE_PATH, f"imgs/structures/{self.object_name}")
        structure = Structure()
        structure.add_blocks(self.blocks)
        structure.place_blocks(positions)
        self.structure = structure
        save_to_json(structure.get_json(), f"{structure_dir}/{self.object_name}.json")
        return structure
    def make_urdf_and_view(self, manipulator_yaml_path: str):
        """
        Thay thế cho make_structure.
        Hàm này phân tích self.positions, tự động tìm base, tạo URDF và hiển thị.
        """

        # devide blocks into 2 groups: joints and non-joints
        joints = []
        non_joints = []

        for part in self.positions:
            part['position_np'] = normalize_position(part['position'])
            if part.get('shape', '').lower() == 'joint':
                joints.append(part)
            else:
                non_joints.append(part)
        
        if not non_joints or not joints:
            print("Need at least one joint and one non-joint part to create a URDF.")
            return

        # automatic find base link
        base_popularity = {part['name']: 0 for part in non_joints}
        for joint in joints:
            distances = [(np.linalg.norm(joint['position_np'] - nj['position_np']), nj['name']) for nj in non_joints]
            closest_base_name = min(distances, key=lambda x: x[0])[1]
            base_popularity[closest_base_name] += 1
            print(f"Joint '{joint['name']}' closest to base '{closest_base_name}' with distance {distances[0][0]:.2f} m")
            print(f"Joint '{joint['name']}' position: {joint['position_np']} m")
            print(f"Base '{closest_base_name}' position: {next(part for part in non_joints if part['name'] == closest_base_name)['position_np']} m")
        
        base_name = max(base_popularity, key=base_popularity.get)
        base_block = next(part for part in non_joints if part['name'] == base_name)
        base_position_np = base_block['position_np']
        print(f"✅ Base Link được xác định là: '{base_name}'")

        # # 3. Tạo định nghĩa Base Link cho SOMO
        base_shape_from_llm = base_block['shape']
        somo_shape = "box" if base_shape_from_llm.lower() == "cuboid" else base_shape_from_llm
        dims_dict = base_block['dimensions']
        print(dims_dict)
        
        if somo_shape == "box":
            somo_dims = [
                dims_dict.get('x', 25) / 1000.0,
                dims_dict.get('y', 25) / 1000.0,
                dims_dict.get('z', 25) / 1000.0,
            ]
        else: # Mặc định cho cylinder
            somo_dims = [
                dims_dict.get('height', 25) / 1000.0,
                dims_dict.get('radius', 25) / 1000.0,
            ]

        somo_base_link = SMLinkDefinition(
            shape_type=somo_shape,
            dimensions=somo_dims, # <--- Dùng kích thước đã được xử lý đúng
            mass=1.0,
            material_color=base_block['color'],
            inertial_values=[0.1, 0, 0, 0.1, 0, 0.1],
            material_name="base_color",
        )

        # 4. Tạo các cặp Manipulator và Offset
        manipulator_def = SMManipulatorDefinition.from_file(manipulator_yaml_path)
        manipulator_pairs = []
        for joint in joints:
            relative_pos_m = (joint['position_np'] - base_position_np) / 1000.0
            angle = np.arctan2(relative_pos_m[1], relative_pos_m[0])
            offset = [relative_pos_m[0], relative_pos_m[1], 0.1, 0, np.pi/2, angle]
            manipulator_pairs.append((manipulator_def, offset))

        # 5. Tạo và lưu file URDF
        os.makedirs(self.structure_dir, exist_ok=True)
        assembly_name = self.object_name
        temp_urdf_file = create_cmassembly_urdf(
            base_links=[somo_base_link],
            manipulator_definition_pairs=manipulator_pairs,
            assembly_name=assembly_name
        )
        self.urdf_path = os.path.join(self.structure_dir, f"{assembly_name}.urdf")
        shutil.copy(temp_urdf_file, self.urdf_path)
        print(f"🎉 URDF đã được tạo tại: {self.urdf_path}")

        # Hiển thị trong PyBullet
        user_choice = input("Bạn có muốn xem model vừa tạo trong PyBullet không? (y/n): ")
        if user_choice.lower() == 'y':
            self._view_urdf_in_pybullet()

    def _view_urdf_in_pybullet(self):
        """Hàm nội bộ để hiển thị file URDF đã tạo."""
        if not self.urdf_path:
            print("❌ Không có file URDF để hiển thị.")
            return

        print(f"👀 Đang mở PyBullet để xem '{os.path.basename(self.urdf_path)}'...")
        client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        p.loadURDF("plane.urdf")
        p.loadURDF(self.urdf_path, basePosition=[0, 0, 0], useFixedBase=True)
        
        print("✅ Tải model thành công. Đóng cửa sổ PyBullet để thoát.")
        for _ in range(1000):  # Giữ cửa sổ mở trong một thời gian ngắn
            p.stepSimulation()
            time.sleep(0.01)
        print("👋 Đã đóng PyBullet.")
    def get_structure_image(self):
        """
        Get the image of the structure.
        """
        structure_dir = os.path.join(BASE_PATH, "imgs/structures")
        isometric_img = get_imgs(keys=["isometric"], axes=True, labels=False)
        img = Image.fromarray(isometric_img)
        img.save(
            f"{structure_dir}/{self.object_name}/{self.object_name}_isometric.png"
        )
        self.structure_image = img
        return img
    def stability_check(self, blocks, debug=False):
        for i in range(len(blocks)):
            structure = Structure()
            structure.add_blocks(blocks[: i + 1])
            structure.place_blocks()

            last_block = blocks[i]

            x_img, y_img = get_imgs(
                keys=["x", "y"], axes=False, labels=False, highlight_id=last_block.id
            )

            stable, pos_delta, rot_delta = structure.check_stability(
                blocks[i].id, debug=debug
            )

            pos_delta = 1000 * np.array(pos_delta)

            if not stable:
                return False, last_block, pos_delta, x_img, y_img

        return True, None, None, x_img, y_img

    def refine_structure(self, blocks):
        for i in range(2):
            stable, unstable_block, pos_delta, x_img, y_img = self.stability_check(
                blocks, debug=True
            )
            if stable:
                break
            else:
                print(unstable_block.gpt_name)
            
            response, stability_context = self.get_stability_correction(
                self.object_name, unstable_block, pos_delta, self.positions, x_img, y_img
            )
            json_output = get_last_json_as_dict(response)
            blocks = blocks_from_json(json_output)
            structure = Structure()
            structure.add_blocks(blocks)
            structure.place_blocks()
            isometric_img = get_imgs(keys=["isometric"], axes=True, labels=False)
            img = Image.fromarray(isometric_img)
            img.save(
                f"{self.structure_dir}/{self.object_name}_stability_correction_{i}.png"
            )
        return stable
    def save_structure(self):
        """
        Save the structure to a JSON file.
        """
        assembly = Assembly(
            structure=self.structure,
            structure_directory=self.structure_dir,
            to_build=self.object_name,
            isometric_image=self.structure_image,
            available_blocks_json=self.available_blocks,
            assembly_num=0,
            eval_rating=None,
            eval_guesses=None,
        )
        assembly.save_to_structure_dir()
        return assembly

    def get_structure_rating(self, iter=0):
        prompt = f"""
        Given your description of the block structure, how well does the structure in the image use blocks to resemble a {self.object_name} considering that the structure is made from  a limited set of toy blocks? Rate the resemblance of the block structure to a {self.object_name} on a scale of 1 to 5 defined by the following:
            1 - the structure in the image has no resemblance to the intended structure. It's missing all key features and appears incoherent
            2 - the structure in the image has a small amount of resemblance to the intented structure. It has at least 1 key feature and shows an attempt at the intended structure
            3 - the structure has clear similarities to the intended structure and appears coherent. It has at least 1 key feature and shows similarities in other ways as well.
            4 - the structure represents multiple key features of the intended design and shows a decent overall resemblance.
            5 - the structure in the image is a good block reconstruction of the intended structure, representing multiple key features and showing an overall resemblance to the intended structure.

        Provide a brief explanation of your though process then provide your final response as JSON in the following format:
        {markdown_json({"rating": 5})}
        """
        response, updated_context = prompt_with_caching(
            prompt,
            self.eval_llm_context,
            self.structure_dir,
            "structure_rating",
            cache=True,
            i=iter,
        )
        self.eval_llm_context = updated_context
        return response

    def get_structure_info(self, iter=0):
        self.eval_llm_context = []
        prompt = [
            self.structure_image,
            f"""
        I am currently building a structure made out of toy blocks shown in the given image. Describe in detail as much as you can about this image. Please list the top 10 things that the structure most resembles in order of similarity.

        After describing the image in detail and providing some initial thoughts, answer as JSON in the following format providing 10 guesses. Your guesses should mostly be single words like "bottle" and never use adjectives like "toy_bottle". 
        {
            markdown_json({"guesses": ["guess_1", "guess_2", "guess_3", "guess_4", "guess_5", "guess_6", "guess_7", "guess_8", "guess_9", "guess_10"]})
        }
        """.strip(),
        ]
        response, updated_context = prompt_with_caching(
            prompt,
            self.eval_llm_context,
            self.structure_dir,
            "structure_info",
            cache=True,
            i=iter,
        )
        self.eval_llm_context = updated_context
        return response
    
    
    
if __name__ == "__main__":
    # Initialize pybullet
    # if not p.isConnected():
    #             p.connect(p.GUI)
    # p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # Pipeline
    isometric_image = IsometricImageJoint(object_name="Octopus")
    description = isometric_image.describe_object()
    plan = isometric_image.make_plan(description)
    order = isometric_image.order_blocks(plan)
    positions = isometric_image.decide_position(order)

    isometric_image.make_urdf_and_view(manipulator_yaml_path='/Users/pqnhhh/Documents/GitHub/multi-agent-block-desgin/data/joint_def.yaml')
    final_urdf_path = isometric_image.urdf_path
    # isometric_image.refine_structure(isometric_image.blocks)
    # isometric_image.get_structure_image()

    # # Save the structure
    # isometric_image.save_structure()

    # # Get the structure info
    # info = isometric_image.get_structure_info()
    # print(info)
    # rating = isometric_image.get_structure_rating()
    # print(rating)
    
    # p.disconnect()