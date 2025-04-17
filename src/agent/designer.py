import os
import sys

# Thêm thư mục gốc của dự án vào sys.path
BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_PATH)
import time
from src.agent.llm import OpenAILLM
from src.prompt import designer_prompt_v2
import matplotlib.pyplot as plt
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union, cast
import torch
import base64
from PIL import Image
import json

AVAILABLE_BLOCKS = os.path.join(BASE_PATH, 'data/simulated_blocks.json')
class Designer:
    def __init__(self, designer_prompt: designer_prompt_v2, llm: OpenAILLM, model_name="gpt-4o", max_tokens=1000):
        super().__init__()
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.designer_prompt = designer_prompt
        self.example_designer = designer_prompt.EXAMPLES_PLANNER
        self.plan = None
        self.thought = None
        self.llm = llm
        self.available_blocks = AVAILABLE_BLOCKS

        with open(self.available_blocks, 'r') as file:
            available_blocks_data = json.load(file)
        self.available_blocks = available_blocks_data

    
    def __call__(self, query: str, previous_plan: str, observation: str, **kwargs) -> Any:
        designer_prompt = self.designer_prompt.PLAN.format(user_query=query, examples=self.example_designer, 
                                                            planning=previous_plan, observer_output=observation, available_blocks=self.available_blocks)
        input_message_planner = [
            {"role": "system", "content": self.llm.system_prompt},
            {"role": "user", "content": designer_prompt},
        ]
        response = self.llm.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=input_message_planner,
            max_tokens=1000,
            stop=["\"\"\""])

        thought_plan = response.choices[0].message.content.strip()
        thought = thought_plan.split('<thought>')[1].split('</thought>')[0].strip()
        plan =  thought_plan.split('<plan>')[1].split('</plan>')[0].strip()

        return thought, plan
    
# if __name__ == "__main__":
#     designer = Designer(designer_prompt_v2, OpenAILLM(api_file="api_key.txt"))
#     print(designer("Design a letter U", "No feedback available.", "No observation available."))