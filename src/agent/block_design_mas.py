import os
import sys
from pathlib import Path

BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_PATH)

from src.agent.llm import OpenAILLM
import matplotlib.pyplot as plt
import numpy as np
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union, cast
import base64
from PIL import Image
from toolset import IsometricImage
from src.agent.designer import Designer
from src.agent.observer import Observer
from src.agent.coder import Coder
from src.prompt import coder_prompt_v3, designer_prompt_v2, observer_prompt_v2
import torch
import json
from src.utils.utils import print_dict, slugify
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

available_blocks = os.path.join(BASE_PATH, 'data/simulated_blocks.json')

class BlockDesignMAS:
    def __init__(self, api_file, max_round=5):
        super().__init__()
        self.llm = OpenAILLM(api_file)
        self.designer = Designer(designer_prompt_v2, self.llm, available_blocks)
        self.coder = Coder(coder_prompt_v3, self.llm)
        self.observer = Observer(observer_prompt_v2, self.llm)
        self.plan = None
        self.observation = None
        self.code = None
        self.max_round = max_round

    def query(self, query: str, positions = None, structure_img = None) -> Any:
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
                out = execute_command(object_name, positions, structure_img)
            except Exception as e:
                out = str(e)
                print("Error:", out)

            if isinstance(out, dict) and "positions" in out:
                # Lấy giá trị từ key 'positions'
                positions_data = out["positions"]
                if positions_data is not None:
                    file_path = f'final_results/positions/{object_name}_result.json'
                    with open(file_path, 'w') as f:
                        json.dump(positions_data, f, indent=4)
                    print(f"Output saved to {file_path}")
                    positions = file_path
            if isinstance(out, dict) and 'image' in out and isinstance(out['image'], Image.Image):
                out['image'] = f"Image object was created successfully."
            self.observation = self.observer(out)
            print(5*'-',"Observation", 5*'-', '\n' + self.observation)
        
        return out
    
if __name__ == "__main__":
    block_agent = BlockDesignMAS(api_file="api_key.txt")
    block_agent.query("Tree", positions='/Users/pqnhhh/Documents/GitHub/multi-agent-block-desgin/final_results/positions/tree_result.json')
    