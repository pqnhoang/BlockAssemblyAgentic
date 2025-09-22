CODE = '''
** Role **: You are an expert software programmer. Your task is to convert the given step-by-step plan into executable Python code using the IsometricImage class.

** Important instructions **:
1. You will be given assembly instructions for building structures with blocks, cylinders, and other objects. Use the IsometricImage class to implement these instructions.
2. Your primary responsibility is to translate assembly plans into Python code that creates the specified structures.
3. IsometricImage is a class that simulates the building of a structure, provide functions to describe object, make plan, order blocks, decide position, make structure, get image, check stability, refine structure, and save structure.
4. You can use base Python (comparison, math, etc.) for basic logical operations.
5. ALWAYS return results as a dictionary for consistency and clarity.

Provided Python Functions/Class:

import math
class IsometricImage:
    """
    A class for physics-based simulation of various objects.

    Attributes
    ----------
    object_name : str
        The name of the object to be built.
    image : Image.Image | Path, optional
        The instruction image to be used for guiding the building process.
    feed_back_image : Image.Image | torch.Tensor | np.ndarray | None, optional
        The isometric image of the structure.
    positions : str | dict  
        The positions of the blocks in the structure.
    available_blocks_path : str, optional
        Path to the JSON file containing available blocks. If None, uses default path.
    
    Methods
    -------
    llm_query(question: str) -> str
        Returns the answer to a question asked about the image using the LLM model.
    describe_object(iter: int) -> str
        Describe the object in the image.
    make_plan(description: str, iter: int) -> str
        Make a plan to assemble the object.
    order_blocks(plan: str, iter: int) -> str
        Make a list of blocks to assemble the object.
    decide_position(order: str, iter: int) -> list[dict]
        Decide the position of the blocks to assemble the object.
    refine_structure(blocks: list[dict]) -> None
        Refine the structure to make it stable.
    make_structure(positions: list[dict]) -> None
        Make the structure of the object in simulation follow the order and position.
    get_structure_image() -> Image.Image
        Get the isometric image of the structure.
    stability_check(blocks: list[dict]) -> tuple[bool, dict, dict, Image.Image, Image.Image]
        Check the stability of the structure.
    save_structure() -> None
        Save the structure of the object in simulation.
    get_structure_rating(iter: int) -> str
        Get the rating of the structure.
    get_structure_info(iter: int) -> str
        Get detailed information about the structure.
    """
    def __init__(self, object_name: str, image: Image.Image | Path = None, feed_back_image: Image.Image | torch.Tensor | np.ndarray | None = None, available_blocks_path=None):
        """Initialize an IsometricImage object."""
        pass
        
    def simple_query(self, question: str = None) -> str:
        """Returns the answer to a basic question asked about the image."""
        pass

    def describe_object(self, iter: int = 0) -> str:
        """
        Describe the object in the image.
        
        Example:
        >>> description = obj.describe_object(iter=0)
        """
        pass
    
    def make_plan(self, description: str, iter: int = 0) -> str:
        """
        Make a plan to assemble the object.
        
        Example:
        >>> plan = obj.make_plan(description, iter=0)
        """
        pass

    def order_blocks(self, plan: str, iter: int = 0) -> str:
        """
        Make a list of blocks to assemble the object.
        
        Example:
        >>> order = obj.order_blocks(plan, iter=0)
        """
        pass
    
    def decide_position(self, order: str, iter: int = 0) -> list[dict]:
        """
        Decide the position of the blocks to assemble the object.
        
        Example:
        >>> positions = obj.decide_position(order, iter=0)
        """
        pass
    
    def refine_structure(self, blocks: list[dict]) -> None:
        """
        Refine the structure to make it stable.
        
        Example:
        >>> obj.refine_structure(obj.blocks)
        """
        pass
    
    def make_structure(self, positions: list[dict]) -> None:
        """
        Make the structure of the object in simulation.
        
        Example:
        >>> obj.make_structure(positions)
        """
        pass
    
    def get_structure_image(self) -> Image.Image:
        """
        Get the image of the structure.
        
        Example:
        >>> image = obj.get_structure_image()
        """
        pass
        
    def stability_check(self, blocks: list[dict]) -> tuple[bool, dict, dict, Image.Image, Image.Image]:
        """
        Check the stability of the structure.
        
        Example:
        >>> is_stable, unstable_block, pos_delta, x_img, y_img = obj.stability_check(obj.blocks)
        """
        pass
    
    def save_structure(self) -> None:
        """
        Save the structure of the object in simulation.
        
        Example:
        >>> obj.save_structure()
        """
        pass
    
    def get_structure_rating(self, iter: int = 0) -> str:
        """
        Get the rating of the structure.
        
        Example:
        >>> rating = obj.get_structure_rating(iter=0)
        """
        pass

    def get_structure_info(self, iter: int = 0) -> str:
        """
        Get detailed information about the structure.
        
        Example:
        >>> info = obj.get_structure_info(iter=0)
        """
        pass
    
    
Write a function using Python and the IsometricImage class (above) that could be executed to provide an answer to the query. 


### Examples
{example}

Plan at this step: {plan}
** Expected format output begin with **
def execute_command(object_name, positions, structure_img, instruction_img=None):
'''

EXAMPLES_CODER = '''
### Example 1 
Plan:
Step 1: Get the general description of the table.
Step 2: Plan which blocks to use to represent the table.
Step 3: Get the general description of the table.
Step 4: Plan which blocks to use to represent the table.
A: ```
def execute_command(object_name, positions, structure_img, instruction_img=None):
    # Initialize the structure
    table = IsometricImage(object_name, instruction_img, positions, structure_img)
    # Step 1: Get the general description of the table.
    description = table.describe_object(iter=0)
    # Step 2: Plan which blocks to use to represent the table.
    plan = table.make_plan(description, iter=0)
    # Step 3: Get the general description of the table.
    order = table.order_blocks(plan, iter=0)
    # Step 4: Plan which blocks to use to represent the table.
    block_positions = table.decide_position(order, iter=0)
    return {"positions": block_positions}
```

### Example 2
Plan:
Step 1: Place the blocks in the simulation follow the order and position.
Step 2: Refine the design to make it stable.
Step 3: Get the isometric image of the structure.
Step 4: Save and return the structure.
A: ```
def execute_command(object_name, positions, structure_img, instruction_img=None):
    # Initialize structure
    table = IsometricImage(object_name, instruction_img, positions, structure_img)
    # Step 1: Place the blocks in the simulation follow the order and position.
    table.make_structure(table.positions)
    # Step 2: Refine the structure. MUST use .blocks and capture the return value.
    stable = table.refine_structure(table.blocks)
    # Step 3: Get the isometric image of the structure.
    img = table.get_structure_image()
    # Step 4: Save and return the structure.
    table.save_structure()
    return {"positions": table.positions, "is_stable": stable, "image": img}
```

### Example 3
Plan:
Step 1: Get the general description of the letter U.
Step 2: Plan which blocks to use to represent the letter U.
Step 3: Decide the position of the blocks to assemble the letter U.
Step 4: Return the blocks position.
A: ```
def execute_command(object_name, positions, structure_img, instruction_img=None):
    # Initialize the structure
    letter = IsometricImage(object_name, instruction_img, positions, structure_img)
    # Step 1: Get the general description of the letter U.
    description = letter.describe_object(iter=0)
    # Step 2: Plan which blocks to use to represent the letter U.
    plan = letter.make_plan(description, iter=0)
    # Step 3: Decide the position of the blocks to assemble the letter U.
    order = letter.order_blocks(plan, iter=0)
    block_positions = letter.decide_position(order, iter=0)
    # Step 4: Return the blocks position.
    return {"positions": block_positions}
```

### Example 4
Plan:
Step 1: Place the blocks in the simulation follow the order and position.
Step 2: Refine the structure to make it stable.
Step 3: Get the isometric image of the structure.
Step 4: Save the structure and return success status.
A: ```
def execute_command(object_name, positions, structure_img, instruction_img=None):
    # Initialize the structure
    letter = IsometricImage(object_name, instruction_img, positions, structure_img)
    # Step 1: Place the blocks in the simulation follow the order and position.
    letter.make_structure(letter.positions)
    # Step 2: Refine the structure. MUST use .blocks and capture the return value.
    stable = letter.refine_structure(letter.blocks)
    # Step 3: Get the isometric image of the structure.
    img = letter.get_structure_image()
    # Step 4: Save the structure and return success status.
    letter.save_structure()
    return {"positions": letter.positions, "is_stable": stable, "image": img}
```

### Example 5
Plan:
Step 1: Make the structure from the positions.
Step 2: Get the information of the letter U.
Step 3: Get the rating of the letter U that was built.
Step 4: Save the structure.
A: ```
def execute_command(object_name, positions, structure_img, instruction_img=None):
    # Initialize the structure
    letter = IsometricImage(object_name, instruction_img, positions, structure_img)
    # Step 1: Make the structure from the positions.
    letter.make_structure(letter.positions)
    # Step 2: Get the guesses information of letter U.
    structure_info = letter.get_structure_info(iter=0)
    # Step 2: Get the rating of the structure U that was built.
    rating_info = letter.get_structure_rating(iter=0)
    # Step 3: Save the structure
    letter.save_structure()
    return {"rating": rating_info,"info": structure_info}
```

### Example 6
Plan:
Step 1: Make the structure from the positions.
Step 2: Get the information of the tree.
Step 3: Get the rating of the tree that was built.
Step 4: Save the structure.
A: ```
def execute_command(object_name, positions, structure_img, instruction_img=None):
    # Initialize the structure
    letter = IsometricImage(object_name, instruction_img, positions, structure_img)
    # Step 1: Make the structure from the positions.
    letter.make_structure(letter.positions)
    # Step 2: Get the guesses information of letter U.
    structure_info = letter.get_structure_info(iter=0)
    # Step 2: Get the rating of the structure U that was built.
    rating_info = letter.get_structure_rating(iter=0)
    # Step 3: Save the structure
    letter.save_structure()
    return {"rating": rating_info,"info": structure_info}
```
'''