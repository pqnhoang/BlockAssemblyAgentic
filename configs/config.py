from pydantic import BaseModel, Field
from typing import Optional, List

class PathSettings(BaseModel):
    """
    Settings for the path configuration.
    """
    base_path: str = Field(
        default="/Users/pqnhhh/Documents/GitHub/multi-agent-block-desgin",
        description="Base path for the project."
    )
    save_dir: str = Field(
        default="/Users/pqnhhh/Documents/GitHub/multi-agent-block-desgin/assets/gpt_caching",
        description="Path to the directory where results will be saved."
    )
    position_dir: str = Field(
        default="/Users/pqnhhh/Documents/GitHub/multi-agent-block-desgin/assets/final_results/positions",
        description="Path to the directory where position results will be saved."
    )
    querry_img_path: str = Field(
        default="/Users/pqnhhh/Documents/GitHub/multi-agent-block-desgin/assets/imgs/block.png",
        description="Path to the directory where query images will be saved."
    )
    data_path: str = Field(
        default="/Users/pqnhhh/Documents/GitHub/multi-agent-block-desgin/assets/data/simulated_blocks_joint.json",
        description="Path to the directory where data will be stored."
    )
    joint_def_path: str = Field(
        default="/Users/pqnhhh/Documents/GitHub/multi-agent-block-desgin/assets/data/joint_def.yaml",
        description="Path to the directory where joint definitions will be stored."
    )
    class Config:
        env_prefix = "path_"
        case_sensitive = False
class LLMSettings(BaseModel):
    """
    Settings for the LLM configuration.
    """
    model_name: str = Field(
        default="gpt-3.5-turbo",
        description="Name of the LLM model to be used."
    )
    max_tokens: int = Field(
        default=4096,
        description="Maximum number of tokens for the LLM response."
    )
    temperature: float = Field(
        default=0.5,
        description="Temperature setting for the LLM response."
    )
    top_p: float = Field(
        default=1.0,
        description="Top-p setting for the LLM response."
    )
class BlockMASSettings(BaseModel):
    path: PathSettings = PathSettings()
    llm: LLMSettings = LLMSettings()