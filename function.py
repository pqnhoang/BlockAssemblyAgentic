import os
import base64
import torch
from torchvision import transforms
from openai import OpenAI
from PIL import Image
import genesis as gs
import numpy as np


BASE_PATH = os.path.dirname(os.path.realpath(__file__))
query_image_path = os.path.join(BASE_PATH, "imgs/query_image.png")
device = "cuda:0" if torch.cuda.is_available() else "cpu"

## OpenAI API
api_file = os.path.join(BASE_PATH, 'api.key')
with open(api_file) as f:
    api_key = f.readline().splitlines()
OPENAI_CLIENT = OpenAI(api_key=api_key[0])
EMBEDDING_MODEL = "text-embedding-ada-002"

class Structure:
    """
    A class to represent a building structure for block assembly.
    """
    def __init__(self, name: str, dimensions: dict, shape: str, number_available: int, original_img):
        if isinstance(image, Image.Image):
            image = transforms.ToTensor()(image)
        elif isinstance(image, np.ndarray):
            image = torch.tensor(image).permute(2, 0, 1)
        elif isinstance(image, torch.Tensor) and image.dtype == torch.uint8:
            image = image / 255
        
        self.name = name
        self.dimensions = dimensions
        self.shape = shape
        self.number_available = number_available
        self.original_img = original_img


    def llm_query(self, question, context=None, long_answer=True, queues=None):
        """Answers a text question using GPT-4o-mini. The input question is always a formatted string with a variable in it.

        Parameters
        ----------
        query: str
            the text question to ask. Must not contain any reference to 'the image' or 'the photo', etc.
        """
        # query = question.format(object_name)
        # self.PIL_img.save(img_path,"PNG")
        self.original_img.save(query_image_path,"PNG")
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
    
    def get_structure_image(self):
        pass
    
    def get_available_blocks(self):
        pass
    
    def check_stability(self):
        pass
    
    def output_structure(self):
        pass
    
    def bool_to_yesno(self, bool_answer: bool):
        return "Yes" if bool_answer else "No"
    
    def get_structure_image(self):
        pass
    
    def get_available_blocks(self):
        pass