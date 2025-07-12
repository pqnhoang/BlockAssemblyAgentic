import matplotlib.pyplot as plt
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union, cast
import torch
import base64
from PIL import Image
from src.agent.llm import OpenAILLM
from pathlib import Path
import json
from src.prompt import observer_prompt_v2

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

class Observer:
    def __init__(self, observer_prompt: observer_prompt_v2, llm: OpenAILLM, model_name="gpt-4o", max_tokens=1000):
        super().__init__()
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.observer_prompt = observer_prompt
        self.plan = None
        self.thought = None
        self.llm = llm

    def __call__(self, execute_results: Union[Dict, str]) -> Any:
        if isinstance(execute_results, dict):
            results_str = json.dumps(execute_results, indent=2)
        else:
            results_str = execute_results

        observer_prompt_text = self.observer_prompt.OBSERVER.replace("{results}", results_str)
        
        input_message_observer = [
            {"role": "system", "content": self.llm.system_prompt},
            {"role": "user", "content": observer_prompt_text},
        ]

        response = self.llm.client.chat.completions.create(
            model=self.model_name,
            messages=input_message_observer,
            max_tokens=self.max_tokens,
            stop=["</observation>"]
        )
        
        response_content = response.choices[0].message.content
        if "<observation>" in response_content:
            observation = response_content.split("<observation>")[1].strip()
        else:
            observation = response_content.strip()
            
        return observation