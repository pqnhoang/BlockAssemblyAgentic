import os
import sys

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH)

from agent.llm import OpenAILLM
from prompt import designer_prompt_v2
import matplotlib.pyplot as plt
from typing import Any
import json
from configs import RDMSettings

settings = RDMSettings()

class Designer:
    def __init__(self, designer_prompt: designer_prompt_v2, llm: OpenAILLM, model_name="gpt-4o", max_tokens=1000, available_blocks_path=None):
        super().__init__()
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.designer_prompt = designer_prompt
        self.example_designer = designer_prompt.EXAMPLES_PLANNER
        self.plan = None
        self.thought = None
        self.llm = llm        
        # Load available blocks
        if available_blocks_path is None:
            available_blocks_path = settings.path['data_path']
        with open(available_blocks_path, 'r') as f:
            self.available_blocks = json.load(f)

    
    def __call__(self, query: str, previous_plan: str, observation: str, **kwargs) -> Any:
        designer_prompt = self.designer_prompt.PLAN.format(available_blocks=self.available_blocks, user_query=query, examples=self.example_designer, planning=previous_plan, observer_output=observation)
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