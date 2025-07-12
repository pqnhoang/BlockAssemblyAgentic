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
        default="/Users/pqnhhh/Documents/GitHub/multi-agent-block-desgin/assets/data/simulated_blocks.json",
        description="Path to the directory where data will be stored."
    )
    class Config:
        env_prefix = "path_"
        case_sensitive = False
        

class BlockMASSettings(BaseModel):
    path: PathSettings = PathSettings()