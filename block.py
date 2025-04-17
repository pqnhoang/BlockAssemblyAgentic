from dataclasses import dataclass
from typing import Dict, Union, Literal, Optional, List, Tuple
import json
import os
from pathlib import Path
import math

def yaw_to_quaternion(yaw_degrees: float) -> Tuple[float, float, float, float]:
    """Convert yaw angle in degrees to quaternion (w, x, y, z)"""
    yaw_rad = math.radians(yaw_degrees)
    w = math.cos(yaw_rad / 2)
    x = 0
    y = 0
    z = math.sin(yaw_rad / 2)
    return (w, x, y, z)

@dataclass
class Dimensions:
    x: Optional[float] = None
    y: Optional[float] = None
    z: Optional[float] = None
    radius: Optional[float] = None
    height: Optional[float] = None

    def get_volume(self) -> float:
        """Calculate the volume of the block based on its shape"""
        if self.radius is not None and self.height is not None:
            # Cylinder volume
            return 3.14159 * (self.radius ** 2) * self.height
        elif all(v is not None for v in [self.x, self.y, self.z]):
            # Cuboid volume
            return self.x * self.y * self.z
        return 0.0

@dataclass
class Position:
    x: float
    y: float
    z: Optional[float] = None

@dataclass
class LLMBlock:
    name: str
    shape: Literal["cuboid", "cylinder"]
    dimensions: Dict[str, float]
    color: List[float]
    position: Position
    orientation: Tuple[float, float, float, float]  # quaternion (w, x, y, z)

    def to_dict(self) -> Dict:
        """Convert the block to a dictionary"""
        position_dict = {"x": self.position.x, "y": self.position.y}
        if self.position.z is not None:
            position_dict["z"] = self.position.z
            
        return {
            "name": self.name,
            "shape": self.shape,
            "dimensions": self.dimensions,
            "color": self.color,
            "position": position_dict,
            "orientation": self.orientation
        }

class Block:
    def __init__(self, name: str, dimensions: Dict[str, float], shape: Literal["cuboid", "cylinder"], number_available: int):
        self.name = name
        self.dimensions = Dimensions(**dimensions)
        self.shape = shape
        self.number_available = number_available
        self.used_count = 0

    def can_use(self) -> bool:
        """Check if the block is still available for use"""
        return self.used_count < self.number_available

    def use(self) -> bool:
        """Mark the block as used if available"""
        if self.can_use():
            self.used_count += 1
            return True
        return False

    def release(self) -> None:
        """Release a used block"""
        if self.used_count > 0:
            self.used_count -= 1

    def get_volume(self) -> float:
        """Get the volume of the block"""
        return self.dimensions.get_volume()

    def get_dimensions(self) -> Dict[str, float]:
        """Get the dimensions of the block"""
        return {k: v for k, v in self.dimensions.__dict__.items() if v is not None}

    def __str__(self) -> str:
        """String representation of the block"""
        return f"Block(name='{self.name}', shape='{self.shape}', available={self.number_available - self.used_count}/{self.number_available})"

class BlockFactory:
    @staticmethod
    def create_block(name: str, data: Dict) -> Block:
        """Factory method to create a block from data"""
        return Block(
            name=name,
            dimensions=data["dimensions"],
            shape=data["shape"],
            number_available=data["number_available"]
        )

    @staticmethod
    def load_blocks_from_json(file_path: str = "data/simulated_blocks.json") -> Dict[str, Block]:
        """Load blocks from a JSON file"""
        # Get the absolute path to the project root
        current_file = Path(__file__)
        project_root = current_file.parent
        json_path = project_root / file_path
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        return {
            name: BlockFactory.create_block(name, block_data)
            for name, block_data in data.items()
        }

    @staticmethod
    def get_block_by_name(blocks: Dict[str, Block], name: str) -> Optional[Block]:
        """Get a block by its name"""
        return blocks.get(name)

    @staticmethod
    def create_llm_block(data: Dict) -> LLMBlock:
        """Create an LLMBlock from data"""
        yaw = data.get("yaw", 0)  # Default to 0 if yaw not provided
        quaternion = yaw_to_quaternion(yaw)
        
        return LLMBlock(
            name=data["name"],
            shape=data["shape"],
            dimensions=data["dimensions"],
            color=data["color"],
            position=Position(**data["position"]),
            orientation=quaternion
        )

    @staticmethod
    def load_llm_blocks_from_json(json_data: List[Dict]) -> List[LLMBlock]:
        """Load LLM blocks from JSON data"""
        return [BlockFactory.create_llm_block(block_data) for block_data in json_data]

if __name__ == "__main__":
    # Test the block system
    blocks = BlockFactory.load_blocks_from_json()
    print("Available blocks:")
    for name, block in blocks.items():
        print(f"- {block}")
        print(f"  Dimensions: {block.get_dimensions()}")
        print(f"  Volume: {block.get_volume():.6f}")

    # Test LLM block creation
    llm_blocks_data = [
        {
            "name": "support1",
            "shape": "cylinder",
            "dimensions": {"radius": 20, "height": 40},
            "color": [0.5, 0.5, 0.5, 1],
            "position": {"x": -50, "y": 0},
            "yaw": 45,  # 45 degrees
        }
    ]
    llm_blocks = BlockFactory.load_llm_blocks_from_json(llm_blocks_data)
    print("\nLLM Blocks:")
    for block in llm_blocks:
        print(f"- {block.name}: {block.to_dict()}") 