import json
import os
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union, cast
from openai import AzureOpenAI, OpenAI
import base64
from pathlib import Path

class OpenAILLM:
    def __init__(self, api_file):
        """
        Initialize OpenAI LLM with API key from file or environment variable.
        
        Args:
            api_file (str, optional): Path to file containing API key
            system_prompt (str): System prompt for the LLM
        """
        # Try to get API key from file first
        if api_file and os.path.exists(api_file):
            with open(api_file) as f:
                self.api_key = f.read().strip()
        else:
            # Fall back to environment variable
            self.api_key = os.getenv("OPENAI_API_KEY")
            
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. Please provide it either through:\n"
                "1. api_file parameter pointing to a file containing the key\n"
                "2. OPENAI_API_KEY environment variable"
            )
            
        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key)
        self.system_prompt = "Please answer follows retrictly the provide format."
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass
# test the llm
# if __name__ == "__main__":
#     llm = OpenAILLM(api_file="api_key.txt")
#     response = llm.client.chat.completions.create(
#         model="gpt-4o",
#         messages=[
#             {"role": "system", "content": llm.system_prompt},
#             {"role": "user", "content": "What is the capital of France?"}
#         ]
#     )
#     print(response.choices[0].message.content)