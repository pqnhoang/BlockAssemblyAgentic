import os
import sys
from pathlib import Path

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH)

from agent.llm import OpenAILLM
import base64
from PIL import Image
import json
from typing import Any
from agent.designer import Designer
from agent.observer import Observer
from agent.coder import Coder
from prompt import coder_prompt_v3, designer_prompt_v2, observer_prompt_v2
from utils import slugify
from configs import RDMSettings
from toolset import IsometricImage  

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
setting = RDMSettings()

class BlockDesignMAS:
    def __init__(self, api_file, max_round=5):
        super().__init__()
        self.llm = OpenAILLM(api_file)
        self.designer = Designer(designer_prompt_v2, self.llm, setting.path.data_path)
        self.coder = Coder(coder_prompt_v3, self.llm)
        self.observer = Observer(observer_prompt_v2, self.llm)
        self.plan = None
        self.observation = None
        self.code = None
        self.max_round = max_round

    def query(self, query: str, positions=None, structure_img=None, instruction_img=None) -> Any:
        object_name = slugify(query)
        for idx in range(self.max_round):
            print(10*'=',f'Round {idx}',10*'=')
            ## Planner
            self.thought, self.plan = self.designer(query=query, previous_plan=self.plan, observation=self.observation)
            print(5*'-',"Thought", 5*'-', '\n'+self.thought)
            print(5*'-',"Plan", 5*'-', '\n'+self.plan)
            if "return to user" in self.plan.lower():
                break
            ## Coder
            self.code = self.coder(self.plan)
            print(5*'-',"Code", 5*'-', '\n' + self.code)
            ## Execute
            print("Executing code...")
            exec(self.code, globals())
            try:
                out = execute_command(object_name, positions, structure_img, instruction_img)
                print("Output:", out)
            except Exception as e:
                out = str(e)
                print("Error:", out)

            if isinstance(out, dict) and "positions" in out:
                positions_data = out["positions"]
                if positions_data is not None:
                    os.makedirs(f'outputs/imgs/structures/{object_name}', exist_ok=True)
                    file_path = f'outputs/imgs/structures/{object_name}/{object_name}.json'
                    with open(file_path, 'w') as f:
                        json.dump(positions_data, f, indent=4)
                    print(f"Output saved to {file_path}")
                    positions = file_path
                    
            if isinstance(out, dict) and 'image' in out and isinstance(out['image'], Image.Image):
                out['image'] = f"Image object was created successfully."
                os.makedirs(f'outputs/imgs/structures/{object_name}', exist_ok=True)
                structure_img = Path(f'outputs/imgs/structures/{object_name}/{object_name}_isometric.png')
                
            self.observation = self.observer(out)
            print(5*'-',"Observation", 5*'-', '\n' + self.observation)
        
        return out

    