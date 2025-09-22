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
import base64
from openai import OpenAI

from structure import (Assembly, 
                        Block, 
                        Structure,
                        blocks_from_json,
                        blocks_from_json_joint,
                        process_available_blocks)
from utils import (get_last_json_as_dict, 
                       load_from_json, 
                       save_to_json, 
                       markdown_json, 
                       slugify)
from prompt.skill_prompt import (
                        get_describe_prompt,
                        get_plan_prompt,
                        get_order_prompt,
                        get_position_prompt,
                        get_stability_prompt,
                        get_rating_prompt,
                        get_info_prompt
)
from pybullet_utils import get_imgs
from prompt import prompt_with_caching
from configs.config import RDMSettings
# Set up paths
BASE_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(BASE_PATH)
query_image_path = os.path.join(BASE_PATH, "assets/imgs/query_image.png")
device = "cuda:0" if torch.cuda.is_available() else "cpu"

## OpenAI API - Use GPTClient for better API key management
from utils.gpt_client import GPTClient
OPENAI_CLIENT = GPTClient._get_instance()
EMBEDDING_MODEL = "text-embedding-ada-002"
settings = RDMSettings()


class IsometricImage:
    """
    A class to represent a isometric image of a object.
    """
    def __init__(self, command_text: str, 
                 image: Image.Image | Path = None,
                 positions: Dict | Path = None, 
                 structure_img: Image.Image | Path = None
                 ):
        """ 
        Initialize a IsometricImage class object by providing the query and feed back image. Then we create a scene object to perform build-in functions.
        Parameters
        ----------
        command_text : str
            The command text to be used for guiding the building process.
        image : Image.Image | Path
            The instruction image to be used for guiding the building process. 
        positions : json
            An json contains postions of building blocks. If None, the positions will be set to None.
        structure_img : Image.Image | Path, optional
            The structure image to be used for guiding the building process.
            
        """
        self.object_name = None
        self.command_text = command_text
        self.blocks = []
        self.structure = None
        self.structure_img = None
        self.main_llm_context: List[Dict] = []
        self.eval_llm_context: List[Dict] = []
        self.available_blocks = process_available_blocks(load_from_json(settings.path['data_path']))
        self.structure_dir = os.path.join(settings.path['base_path'], f"{settings.path['save_dir']}/{self.object_name}")
        # Load instruction image
        if isinstance(image, Image.Image) or isinstance(image, np.ndarray):
            self.instruct_img = image
        elif isinstance(image, (Path, str)):
            self.instruct_img = Image.open(image)
        else:
            self.instruct_img = None
        # Load positions
        if isinstance(positions, str):
            with open(positions, 'r') as positions_data:
                self.positions = json.load(positions_data)
        elif isinstance(positions, dict):
            self.positions = positions
        else:
            self.positions = None
        
        # Load blocks
        if self.positions is not None:
            self.blocks = blocks_from_json_joint(self.positions)
        
    def object_name_extract(self, command_text: str) -> str:
        """Returns the object name from the command text.
        Differs from 'exists' in that it presupposes the existence of the object specified by object_name, instead
        checking whether the object possesses the property.
        Parameters
        -------
        command_text : str
            A string describing the name of the object to be found in the image.
        """
        
        prompt = f"What is the object the user want to create in {command_text}? Answer with only the object name."

        ## GPT-4o-mini
        self.instruct_img.save(query_image_path,"PNG")
        with open(query_image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        response = OPENAI_CLIENT.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}",
                                        "detail": "low",
                            },
                        },
                    ],
                }
            ],
        )
        answer = response.choices[0].message.content
        out = slugify(answer)
        self.object_name = out
        return out
    
    def describe_object(self, iter=0):
        """
        Describe the object in the image.
        """

        prompt = get_describe_prompt(self.object_name)
        self.main_llm_context = []
        message_and_images = [prompt] if self.instruct_img is None else [self.instruct_img, prompt]
        response, updated_context = prompt_with_caching(
            message_and_images,
            self.main_llm_context,
            self.structure_dir,
            "description",
            cache=True,
            temeprature=settings.llm['temperature'],
            i=iter,
    )
        self.main_llm_context = updated_context
        return response
    
    def make_plan(self, description, iter=0):
        """
        Make a plan to assemble the object.
        """
        prompt = get_plan_prompt(self.object_name, description, self.available_blocks)
        response,updated_context = prompt_with_caching(
            prompt,
            self.main_llm_context,
            self.structure_dir,
            "main_plan",
            cache=True,
            temeprature=settings.llm['temperature'],
            i=iter,
    )
        self.main_llm_context = updated_context
        return response
    def order_blocks(self, plan, iter=0):
        """
        Order the blocks in the plan.
        """
        prompt = get_order_prompt(self.object_name, plan)
        response, updated_context = prompt_with_caching(
            prompt,
            self.main_llm_context,
            self.structure_dir,
            "order_plan",
            cache=True,
            temeprature=settings.llm['temperature'],
            i=iter,
    )
        self.main_llm_context = updated_context
        return response

    def decide_position(self, order, iter=0):
        """
        Decide the position of the blocks in the order.
        """
        prompt = get_position_prompt(self.object_name, order)
        response, updated_context = prompt_with_caching(
            prompt,
            self.main_llm_context,
            self.structure_dir,
            "decide_position",
            cache=True,
            temeprature=settings.llm['temperature'],
            i=iter,
        )
        
        self.main_llm_context = updated_context
        json_output = get_last_json_as_dict(response)
        blocks = blocks_from_json_joint(json_output)
        self.blocks = blocks
        self.positions = json_output
        return json_output
    
    def get_stability_correction(self, to_build, unstable_block: Block, pos_delta, structure_json, x_img, y_img, iter=0):
        prompt = get_stability_prompt(to_build, unstable_block, pos_delta, structure_json, x_img, y_img)
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
        structure = Structure(object_name=self.object_name)
        structure.add_blocks(self.blocks)
        structure.place_blocks()
        self.structure = structure
        # Update positions
        self.positions = positions
        save_to_json(structure.get_json(), f"{self.structure_dir}/{self.object_name}.json")
        return structure
    
    def get_structure_image(self):
        """
        Get the image of the structure.
        """
        isometric_img = get_imgs(keys=["isometric"], axes=True, labels=False)
        img = Image.fromarray(isometric_img)
        img.save(
            f"{self.structure_dir}/{self.object_name}_isometric.png"
        )
        self.structure_img = img
        return img
    def stability_check(self, blocks, debug=False):
        x_img = None
        y_img = None
        
        for i in range(len(blocks)):
            structure = Structure(self.object_name)
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
        return True
    def save_structure(self):
        """
        Save the structure to a JSON file.
        """
        assembly = Assembly(
            structure=self.structure,
            structure_directory=self.structure_dir,
            to_build=self.object_name,
            isometric_image=self.structure_img,
            available_blocks_json=self.available_blocks,
            assembly_num=0,
            eval_rating=None,
            eval_guesses=None,
        )
        assembly.save_to_structure_dir()
        return assembly

    def get_structure_rating(self, iter=0):
        prompt = get_rating_prompt(self.object_name)
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
        prompt = get_info_prompt()
        # If no structure image is available, keep only text; otherwise prepend the image
        if isinstance(self.structure_img, (Image.Image, np.ndarray)):
            prompt.insert(0, self.structure_img)
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
    img_path = os.path.join(BASE_PATH, "assets", "giraffe.jpg")
    img = Image.open(img_path)
    isometric_image = IsometricImage(command_text="From the image, create a giraffe", 
                                     image=img, 
                                     positions=None, 
                                     structure_img=None)
    
    isometric_image.object_name_extract(isometric_image.command_text)
    print(isometric_image.object_name)