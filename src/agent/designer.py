import os
import time
from llm import OpenAILLM
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prompt import designer_prompt_v1
import matplotlib.pyplot as plt
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union, cast
import torch
import base64
from PIL import Image

class Designer:
    def __init__(self, designer_prompt: designer_prompt_v1, llm: OpenAILLM, model_name="gpt-4o", max_tokens=1000):
        super().__init__()
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.designer_prompt = designer_prompt
        self.example_designer = designer_prompt.EXAMPLE
        self.plan = None
        self.thought = None
        self.llm = llm
        self.output_format = designer_prompt.OUTPUT_FORMAT
    
    def __call__(self, query: str, previous_plan: str, observation: str, **kwargs) -> Any:
        designer_prompt = self.designer_prompt.PLAN.format(user_query=query, examples=self.example_designer, return_format=self.output_format)

        input_message_planner = [
            {"role": "system", "content": self.llm.system_prompt},
            {"role": "user", "content": designer_prompt},
        ]
                
        response = self.llm.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=input_message_planner,
            max_tokens=1000,
            stop=["\"\"\""])

        # thought_plan = response.choices[0].message.content.strip()
        # thought = thought_plan.split('<thought>')[1].split('</thought>')[0].strip()
        # plan =  thought_plan.split('<plan>')[1].split('</plan>')[0].strip()

        return response.choices[0].message.content.strip()
    
if __name__ == "__main__":
    designer = Designer(designer_prompt_v1, OpenAILLM(api_file="api_key.txt"))
    print(designer("design a chair", "", ""))