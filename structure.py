import os
import base64
import torch
import torchvision
from openai import OpenAI
from PIL import Image
import genesis as gs
import numpy as np
import requests
import json


BASE_PATH = os.path.dirname(os.path.realpath(__file__))
query_image_path = os.path.join(BASE_PATH, "imgs/block.png")
device = "cuda:0" if torch.cuda.is_available() else "cpu"

## OpenAI API
api_file = os.path.join(BASE_PATH, 'api_key.txt')
with open(api_file) as f:
    api_key = f.readline().splitlines()
OPENAI_CLIENT = OpenAI(api_key=api_key[0])
EMBEDDING_MODEL = "text-embedding-ada-002"
###CONSTANT
dt = 5e-4
E, nu = 3.e4, 0.45
rho = 1000.


class Structure:
    """
    A class to represent a building structure for block assembly.
    """
    def __init__(self, target_object: str, available_blocks, shape: str, number_available: int, feed_back_image):
        
        gs.init(seed=0, precision='32', logging_level='debug',backend=gs.cpu)
        
        if isinstance(feed_back_image, Image.Image):
            feed_back_image = torchvision.transforms.ToTensor()(feed_back_image)
        elif isinstance(feed_back_image, np.ndarray):
            feed_back_image = torch.tensor(feed_back_image).permute(2, 0, 1)
        elif isinstance(feed_back_image, torch.Tensor) and feed_back_image.dtype == torch.uint8:
            feed_back_image = feed_back_image / 255
        
        with open(available_blocks, 'r') as file:
          available_blocks_data = json.load(file)
        
        self.target_object = target_object
        self.available_blocks = available_blocks_data
        self.shape = shape
        self.number_available = number_available
        self.feed_back_image = feed_back_image
        self.original_img = torchvision.transforms.ToPILImage()(feed_back_image)

        self.scene = gs.Scene(
            viewer_options = gs.options.ViewerOptions(
                camera_pos    = (0, -1.5, 1.5),
                camera_lookat = (0.0, 0.0, 0),
                camera_fov    = 30, 
                max_FPS       = 60,
            ),
            sim_options = gs.options.SimOptions(
                dt = 0.01,
            ),
            show_viewer=False,
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
        self.scene.add_entity(gs.morphs.Plane())

        self.cam_top = self.scene.add_camera(
            res=(1280, 720),
            pos=(0, 0, 2),  # Move camera back and up
            lookat=(0.0, 0.0, 0),  # Focus on center of structure
            fov=30,  # Slightly wider field of view
            GUI=False,
        )
        self.cam_left = self.scene.add_camera(
            res=(1280, 720),
            pos=(0, 2, 0),  # Move camera back and up
            lookat=(0.0, 0.0, 0),  # Focus on center of structure
            fov=30,  # Slightly wider field of view
            GUI=False,
        )

        
        
    def create_cylinder(self, target_: str, x: float, y: float, z: float):
        cylinder = self.scene.add_entity(
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
        return cylinder
    def create_block(self, target_: str, x: float, y: float, z: float):
        block = self.scene.add_entity(
            gs.morphs.Box(
                size=(0.02, 0.02, 0.04),
                pos=(0.5, 0.1, 0.02),
            ),
            material=gs.materials.Rigid(
                friction=1,
            ),
            surface=gs.surfaces.Default(
                color=(0.5, 0.5, 0.5),
            )
        )
        return block

    def create_soft_sphere(self, target_: str, x: float, y: float, z: float):
      robot_mpm = self.scene.add_entity(
      morph=gs.morphs.Sphere(
          pos=(0.5, 0.2, 0.3),
          radius=0.1,
      ),
      material=gs.materials.MPM.Muscle(
          E=E,
          nu=nu,
          rho=rho,
          model='neohooken',
      ),
      surface=gs.surfaces.Default(
          color=(0.5, 0.5, 0.5),
      )
  )
      return robot_mpm
    
    def create_muscle(self, target_: str, x: float, y: float, z: float):
        pass
    def get_structure_image(self):
    # Render from top camera
      rgb_arr_top, depth_arr_top, seg_arr_top, normal_arr_top = self.cam_top.render(
          rgb=True,
          depth=True, 
          segmentation=True,
          normal=True
      )
      
      # Render from left camera
      rgb_arr_left, depth_arr_left, seg_arr_left, normal_arr_left = self.cam_left.render(
          rgb=True,
          depth=True, 
          segmentation=True,
          normal=True
      )
      
      # Convert NumPy arrays to PIL Images
      # Note: RGB arrays from genesis are already in RGB format, not BGR
      pil_image_top = Image.fromarray(rgb_arr_top)
      pil_image_left = Image.fromarray(rgb_arr_left)
      
      # Optionally save to files
      output_dir = "/Users/pqnhhh/Documents/GitHub/multi-agent-block-desgin/imgs/scene_img"
      os.makedirs(output_dir, exist_ok=True)
      
      pil_image_top.save(os.path.join(output_dir, "rgb_top.png"))
      pil_image_left.save(os.path.join(output_dir, "rgb_left.png"))
      
      # Return the PIL images
      return pil_image_top, pil_image_left
    def contact_detection(self, entity1, entity2):
      contact_info = entity1.get_contacts(with_entity=entity2)
      return contact_info
    def move_until_contact(self, target_: str, x: float, y: float, z: float):
        pass
    
    def check_stability(self, target_: str, x: float, y: float, z: float):
       return True
    def simple_query(self, question: str):
        """Returns the answer to a basic question asked about the image. If no question is provided, returns the answer
        to "What is this?". The questions are about basic perception, and are not meant to be used for complex reasoning
        or external knowledge.
        Parameters
        -------
        question : str
            A string describing the question to be asked.
        """
        prompt = question
        ### BLIP2
#         inputs = processor(images=self.PIL_img, text=prompt, return_tensors="pt").to(device="cuda", dtype=torch.bfloat16)
#         generated_ids = model_blip.generate(**inputs)
#         generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
#         return generated_text

        ### GPT-4o-mini
        self.original_img.save(query_image_path,"PNG")
        with open(query_image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')

        headers = {
          "Content-Type": "application/json",
          "Authorization": f"Bearer {OPENAI_CLIENT.api_key}"
        }

        payload = {
          "model": "gpt-4o-mini",
          "messages": [
            {
              "role": "user",
              "content": [
                {
                  "type": "text",
                  "text": prompt
                },
                {
                  "type": "image_url",
                  "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    "detail": "low"
                  }
                }
              ]
            }
          ],
          "max_tokens": 300
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        response_message = response.json()["choices"][0]["message"]["content"]
        return response_message

    def llm_query(self, question, context=None, long_answer=True, queues=None):
        """Answers a text question using GPT-4o-mini. The input question is always a formatted string with a variable in it.

        Parameters
        ----------
        query: str
            the text question to ask. Must not contain any reference to 'the image' or 'the photo', etc.
        """
        # Use the first image (top view) for the query
        self.original_img[0].save(query_image_path,"PNG")
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

if __name__ == "__main__":
  available_blocks = os.path.join(BASE_PATH, 'data/simulated_blocks.json')
  feed_back_image = Image.open(os.path.join(BASE_PATH, 'imgs/block.png'))
  structure = Structure(target_object="block", available_blocks=available_blocks, shape="", number_available=0, feed_back_image=feed_back_image)
  # cylinder = structure.create_cylinder("cylinder", 0.25, 0.1, 0.02)
  # block = structure.create_block("block", 0.5, 0.1, 0.02)
  # robot_mpm = structure.create_soft_sphere("soft_sphere", 0.5, 0.1, 0.02)
  structure.scene.build()
  # for i in range(10):
  #   actu = np.array([0.2 * (0.5 + np.sin(0.01 * np.pi * i))])
  #   robot_mpm.set_actuation(actu)
  #   structure.scene.step()
  # contact_info = structure.contact_detection(block, cylinder)
  # print(contact_info)
  # print(block.detect_collision(0))

  structure.get_structure_image()
  top_img, left_img = structure.get_structure_image()
  structure.original_img = [top_img, left_img]
  print(structure.llm_query("Does each block is stable or not?"))
