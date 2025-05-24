import os
import sys

# Thêm thư mục gốc của dự án vào sys.path
BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_PATH)

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
        """
        Quan sát kết quả thực thi (chỉ text) và tạo ra một "observation".
        
        Args:
            execute_results (Union[Dict, str]): Kết quả từ Coder agent.
        
        Returns:
            str: Chuỗi observation được trích xuất.
        """
        # 1. Chuyển đổi kết quả thực thi thành chuỗi
        if isinstance(execute_results, dict):
            results_str = json.dumps(execute_results, indent=2)
        else:
            results_str = execute_results

        # 2. SỬA LỖI: Thay thế an toàn bằng .replace() thay vì .format()
        observer_prompt_text = self.observer_prompt.OBSERVER.replace("{results}", results_str)
        
        # 3. Tạo thông điệp (chỉ text) để gửi đến API
        input_message_observer = [
            {"role": "system", "content": self.llm.system_prompt},
            {"role": "user", "content": observer_prompt_text},
        ]

        # 4. Gọi API và nhận kết quả
        response = self.llm.client.chat.completions.create(
            model=self.model_name,
            messages=input_message_observer,
            max_tokens=self.max_tokens,
            stop=["</observation>"]
        )
        
        # 5. Trích xuất nội dung từ bên trong thẻ <observation>
        response_content = response.choices[0].message.content
        if "<observation>" in response_content:
            observation = response_content.split("<observation>")[1].strip()
        else:
            observation = response_content.strip()
            
        return observation
    
if __name__ == "__main__":
    output3 = {
    "info": "The structure shows a minimalist representation of a giraffe with distinct neck and body sections. Top guesses: ['giraffe', 'animal', 'horse', 'llama', 'creature', 'deer', 'camel', 'quadruped', 'sculpture', 'toy']",
    "rating": "Rating: 4/5. The structure successfully captures the key features of a giraffe including the long neck, body, and legs. The proportions are appropriate for a block-based representation."
}
    observer = Observer(observer_prompt_v2, OpenAILLM(api_file="api_key.txt"))
    print(observer(output3))