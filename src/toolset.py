import os
import sys
from PIL import Image
import numpy as np
import json
import pybullet as p
import pybullet_data
from typing import Dict, List
from pathlib import Path
import torch

BASE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(BASE_PATH)
query_image_path = os.path.join(BASE_PATH, "assets/imgs/block.png")
device = "cuda:0" if torch.cuda.is_available() else "cpu"

#DESIGN CLASS
from src.structure import (Assembly, 
                        Block, 
                        Structure,
                        blocks_from_json,
                        blocks_from_json_joint,
                        process_available_blocks)
from src.pybullet_utils import get_imgs
from src.utils import (get_last_json_as_dict, 
                       load_from_json, 
                       save_to_json, 
                       markdown_json, 
                       slugify)
from src.prompt import prompt_with_caching
from configs import BlockMASSettings



settings = BlockMASSettings()

class IsometricImage:
    """
    A class to represent a isometric image of a object.
    """
    def __init__(self, object_name: str, 
                 image: Image.Image | Path = None,
                 positions: Dict | Path = None, 
                 structure_img: Image.Image | Path = None, 
                 available_blocks_path: str = None):
        """
        Initialize a IsometricImage class object by providing the query and feed back image. Then we create a scene object to perform build-in functions.
        Parameters
        ----------
        object_name : str
            The name of the object to be built.
        image : Image.Image | Path, optional
            The instruction image to be used for guiding the building process. 
        positions : json
            An json contains postions of building blocks. If None, the positions will be set to None.
        available_blocks_path : str, optional
            Path to the JSON file containing available blocks. If None, uses default path.
            
        """
        # Normalize/validate the instruction image input. Accept PIL.Image, numpy array, or a valid file path.
        if isinstance(image, Image.Image) or isinstance(image, np.ndarray):
            self.instruct_img = image
        elif isinstance(image, (Path, str)):
            if os.path.exists(image):
                try:
                    self.instruct_img = Image.open(image)
                except Exception:
                    self.instruct_img = None
            else:
                self.instruct_img = None
        else:
            self.instruct_img = None
        self.object_name = slugify(object_name)
        self.structure_dir = os.path.join(settings.path.save_dir, self.object_name)
        
        # Load available blocks
        if available_blocks_path is None:
            available_blocks_path = settings.path.data_path
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
        
        # Initialize context variables
        self.main_llm_context: List[Dict] = []
        self.eval_llm_context: List[Dict] = []
    
    def describe_object(self, iter=0):
        """
        Describe the object in the image.
        """

        prompt = f"""
        I'm working on constructing a block tower that represents a(n) {self.object_name}. 
        Following the image instruction, provide a qualitative description of the design that captures its essence in a minimalistic style. The design should focus on simplicity, avoiding unnecessary complexity while still conveying the key features of a(n) {self.object_name}. The description should highlight the overall structure and proportions, emphasizing how the block arrangement reflects the object's shape and form. However the design shouldn't be too large, too wide, or too tall. The available blocks are not suitable to demonstrate the details of the object (e.g eyes, ears, ...), so you should focus on the overall structure and proportions.
        """.strip()
        self.main_llm_context = []
        # If no instruction image is provided, only send the textual prompt
        message_and_images = [prompt] if self.instruct_img is None else [self.instruct_img, prompt]
        response, updated_context = prompt_with_caching(
            message_and_images,
            self.main_llm_context,
            self.structure_dir,
            "description",
            cache=True,
            temeprature=settings.llm.temperature,
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

        Cones are always positioned with their flat side down and their pointed tip facing upwards. This means the base of the cone lies parallel to the ground plane, with the cone's height extending along the z-axis and the radius along the x and y axes.

        For soft blocks (joint shape), these are flexible blocks that can bend and flex like soft fingers, tentacles, or flexible connectors. They comprise from multiple small block segments connected by joints, allowing them to deform naturally under forces. Use these when you need flexible parts (e.g tails of an animal, tentacles of octopus, flexible supports, ...). They have mass properties that affect stability, so account for their weight in the overall structure balance. Soft blocks can be good connectors between rigid parts or as dynamic elements that move. 
        
        **Important**: To ensure the mass properties, we need to specify the base block that the soft block is connected to, and the joint angles between the segments. The base block is the first block in the soft block's list of blocks, and the joint angles are the angles between each segment in the soft block.

        Decide a semantic name for the block for the role it represents in the structure. 
        Decide the colors of each block to look like a {self.object_name}. Color is an rgba array with values from 0 to 1.
        """
        
        response,updated_context = prompt_with_caching(
            prompt,
            self.main_llm_context,
            self.structure_dir,
            "main_plan",
            cache=True,
            temeprature=settings.llm.temperature,
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
            temeprature=settings.llm.temperature,
            i=iter,
    )
        self.main_llm_context = updated_context
        return response

    def decide_position(self, order, iter=0):
        """
        Decide the position of the blocks in the order.
        """
        
        prompt = f"""
        With the stacking order determined as {order}, I need to know the precise x, y, z positions and yaw, pitch, roll orientation (in degrees) for each block to build a '{self.object_name}' structure.

        General Rules:
        1.  The structure should be stable and centered around the origin (0, 0).
        2.  Blocks at similar heights must be spaced out in x and y to avoid collision.
        3.  Every block must stack correctly on its specified base (or the ground).
        4.  The `yaw` angle is the rotation around the vertical z-axis.

        Block-Specific Instructions:

        **For Rigid Blocks (cuboid, cylinder):**
        -   The `x`, `y` coordinates represent the center of the block.
        -  `pitch` and `roll` angles are always 0.
        -   For rigid blocks, the "baseblock" is always "None" since they are not connected to any other block.

        **For Joint Blocks:**
        -   You must specify the `base_block` that the joint is connected to.
        -   The `x`, `y`, `z` in the 'offset' represent the starting position of the joint's base, relative to the center of its `base_block`.
        -   The `yaw`, `pitch`, and `roll` angles determine the initial orientation of the joint's base. Think of it like attaching a robotic arm to a body.
            -   **Attaching on TOP:** If a joint is placed on the top surface of a horizontal block (like a snake on the floor), its `pitch` and `roll` will typically be 0. The `yaw` will control its direction.
            -   **Attaching to the SIDE:** If a joint needs to stick out horizontally from the side of a block (like an arm on a torso), you MUST use `pitch` or `roll`.
                -   **Example:** A `pitch` of 90 degrees will make the joint point straight out, perpendicular to the vertical side of the `base_block`.
                -   **Example:** A `roll` can then be used to twist the joint's bending axis. A `roll` of 90 degrees would change its bending motion from up-down to left-right.
        - Careful design the position x, y, z. **Crucial Calculation Example:** To attach a block to the **edge** of a cylinder `base_block` that has a `radius` of 20 and is centered at its own (0,0), the attached block's `offset` must have `x: 20` (or `y: 20`) to be perfectly on the edge. An `x` value like 25 would be incorrect and floating in space.
        Provide your reasoning for the chosen positions and orientations for each block.

        Output a JSON following this exact format:
        {markdown_json(
            [
                {
                    "name": "support1",
                    "shape": "cylinder",
                    "dimensions": {"radius": 20, "height": 40},
                    "color": [0.5, 0.5, 0.5, 1],
                    "offset": {"x": -50, "y": 0, "z": 0, "yaw": 0, "pitch": 0, "roll": 0},
                    "base_block": "None"
                },
                {
                    "name": "deck",
                    "shape": "cuboid",
                    "dimensions": {"x": 100, "y": 50, "z": 20},
                    "color": [0.5, 0.5, 0.5, 1],
                    "offset": {"x": 0, "y": 0, "z": 40, "yaw": 0, "pitch": 0, "roll": 0},
                    "base_block": "support1",
                },
                {
                    "name": "tentacle_arm",
                    "shape": "joint",
                    "dimensions": {"x": 100, "y": 10, "z": 10},
                    "color": [0.5, 0.5, 0.5, 1],
                    "offset": {"x": 0, "y": 30, "z": 10, "yaw": 0, "pitch": 90, "roll": 0},
                    "base_block": "deck",
                }
            ]
        )}
        """
        response, updated_context = prompt_with_caching(
            prompt,
            self.main_llm_context,
            self.structure_dir,
            "decide_position",
            cache=True,
            temeprature=settings.llm.temperature,
            i=iter,
        )
        
        self.main_llm_context = updated_context
        json_output = get_last_json_as_dict(response)
        blocks = blocks_from_json_joint(json_output)
        self.blocks = blocks
        # for block in blocks:
        #     print(block.get_json()) 
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
            """
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
        structure_dir = os.path.join(settings.path.base_path, f"./assets/imgs/structures/{self.object_name}")
        structure = Structure(object_name=self.object_name)
        # for block in self.blocks:
        #     print(block.get_json())
        structure.add_blocks(self.blocks)
        structure.place_blocks(positions)
        self.structure = structure
        save_to_json(structure.get_json(), f"{structure_dir}/{self.object_name}.json")
        return structure
    
    def get_structure_image(self):
        """
        Get the image of the structure.
        """
        structure_dir = os.path.join(settings.path.base_path, "assets/imgs/structures")
        isometric_img = get_imgs(keys=["isometric"], axes=True, labels=False)
        img = Image.fromarray(isometric_img)
        img.save(
            f"{structure_dir}/{self.object_name}/{self.object_name}_isometric.png"
        )
        self.structure_image = img
        return img
    def stability_check(self, blocks, debug=False):
        for i in range(len(blocks)):
            structure = Structure(object_name=self.object_name)
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
            
            response, _ = self.get_stability_correction(
                self.object_name, unstable_block, pos_delta, self.positions, x_img, y_img
            )
            json_output = get_last_json_as_dict(response)
            blocks = blocks_from_json_joint(json_output)
            structure = Structure(object_name=self.object_name)
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
            f"""
        I am currently building a structure made out of toy blocks shown in the given image. Describe in detail as much as you can about this image. Please list the top 10 things that the structure most resembles in order of similarity.

        After describing the image in detail and providing some initial thoughts, answer as JSON in the following format providing 10 guesses. Your guesses should mostly be single words like "bottle" and never use adjectives like "toy_bottle". 
        {
            markdown_json({"guesses": ["guess_1", "guess_2", "guess_3", "guess_4", "guess_5", "guess_6", "guess_7", "guess_8", "guess_9", "guess_10"]})
        }
        """.strip(),
        ]
        # If no structure image is available, keep only text; otherwise prepend the image
        if isinstance(self.structure_image, (Image.Image, np.ndarray)):
            prompt.insert(0, self.structure_image)
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
    if not p.isConnected():
                p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    img = Image.open("/Users/pqnhhh/Documents/GitHub/multi-agent-block-desgin/assets/instruction_imgs/octopus.jpg")
    isometric_image = IsometricImage(object_name="Octopus", image=None)
    description = isometric_image.describe_object()
    plan = isometric_image.make_plan(description)
    order = isometric_image.order_blocks(plan)
    positions = isometric_image.decide_position(order)
    isometric_image.make_structure(isometric_image.positions)
    isometric_image.refine_structure(isometric_image.blocks)
    isometric_image.get_structure_image()

    # Save the structure
    isometric_image.save_structure()

    # Get the structure info
    info = isometric_image.get_structure_info()
    rating = isometric_image.get_structure_rating()
    
    p.disconnect()