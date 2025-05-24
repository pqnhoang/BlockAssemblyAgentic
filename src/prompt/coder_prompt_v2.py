CODE = '''
** Role **: You are an expert software programmer. Your task is to convert the given step-by-step plan into executable Python code using the Structure class.

** Important instructions **:
1. You will be given assembly instructions for building structures with blocks, cylinders, and other objects. Use the Structure class to implement these instructions.
2. Your primary responsibility is to translate assembly plans into Python code that creates the specified structures.
3. IsometricImage is a class that simulates the building of a structure, provide a function to create a block, get the structure image, check stability, simple query, llm query, describe object, make plan, order blocks, decide position and build scene.
4. You can use base Python (comparison, math, etc.) for basic logical operations.

Provided Python Functions/Class:

import math
class IsometricImage:
    """
    A class for physics-based simulation of various objects.

    Attributes
    ----------
    object_name : str
            The name of the object to be built.
        feed_back_image : array_like
            An array-like of the cropped image taken from the original image.
        available_blocks_path : str, optional
            Path to the JSON file containing available blocks. If None, uses default path.
    
    Methods
    -------
    simple_query(question: str)->str
        Returns the answer to a basic question asked about the image.
    llm_query(question: str)->str
        Returns the answer to a question asked about the image using the LLM model. Typical use when the question is complex, ambiguous, or requires external knowledge. 
        Typically ask about the object properties, relationships between them. For example: Ask the color of the Kleenex package in the image.
    describe_object(iter: int)->str
        Describe the object in the image.
    make_plan(description: str, iter: int)->str
        Make a plan to assemble the object.
    order_blocks(plan: str, iter: int)->str
        Make a list of blocks to assemble the object.
    decide_position(order: str, iter: int)->list[dict]
        Decide the position of the blocks to assemble the object.
    refine_structure(blocks: list[dict]) -> None
        Refine the structure to make it stable.
    make_structure(positions: list[dict])->None
        Make the structure of the object in simulation follow the order and position.
    get_structure_image()->Image.Image
        Get the isometric image of the structure.
    stability_check() -> tuple[bool, dict, dict, Image.Image, Image.Image]
        Check the stability of the structure.
    save_structure()->None
        Save the structure of the object in simulation.
    """
    def __init__(self, object_name: str, feed_back_image: Image.Image | torch.Tensor | np.ndarray | None, available_blocks_path= None):
        """
        Initialize a Structure class object by providing the object name, available blocks path and feed back image. Then we create a scene object to perform build-in functions.
        Parameters
        ----------
        object_name : str
            A string describing the object name to be created.
        feed_back_image : array_like
            An array-like of the cropped image taken from the original image.
        available_blocks_path : str
            The path to the JSON file containing available blocks.
        """
        self.object_name = slugify(object_name)
        self.structure_dir = os.path.join(SAVE_DIR, self.object_name)

        if feed_back_image is None:
            feed_back_image = Image.open(os.path.join(BASE_PATH, "imgs/block.png"))
        else:
            feed_back_image = feed_back_image
        
        if isinstance(feed_back_image, Image.Image):
            feed_back_image = torchvision.transforms.ToTensor()(feed_back_image)
        elif isinstance(feed_back_image, np.ndarray):
            feed_back_image = torch.tensor(feed_back_image).permute(2, 0, 1)
        elif isinstance(feed_back_image, torch.Tensor) and feed_back_image.dtype == torch.uint8:
            feed_back_image = feed_back_image / 255
        
        self.original_img = torchvision.transforms.ToPILImage()(feed_back_image)
        self.feed_back_image = feed_back_image
        # Load available blocks
        if available_blocks_path is None:
            available_blocks_path = os.path.join(BASE_PATH, "data/simulated_blocks.json")
        available_blocks = load_from_json(available_blocks_path)
        self.available_blocks = process_available_blocks(available_blocks)
        self.blocks = None
        self.positions = None
        
    def simple_query(self, question: str = None) -> str:
        """Returns the answer to a basic question asked about the image. If no question is provided, returns the answer
        to "What is this?". The questions are about basic perception, and are not meant to be used for complex reasoning
        or external knowledge.
        Parameters
        -------
        question : str
            A string describing the question to be asked.

        Examples
        -------

        >>> # How many leg does the giraffe have?
        >>> def execute_command(object_name, feed_back_image) -> str:
        >>>     giraffe = IsometricImage(object_name="Giraffe", feed_back_image=None)
        >>>     return giraffe.simple_query("How many leg does the giraffe have?")

        >>> # What is the color of the tree leaf?
        >>> def execute_command(object_name, feed_back_image) -> str:
        >>>     tree = IsometricImage(object_name="Tree", feed_back_image=None)
        >>>     return tree.simple_query("What is the color of the tree leaf?")

    def describe_object(self, iter: int)->str:
        """
        Describe the object in the image.
        Parameters
        ----------
        iter : int
            The iteration of the description.
        Returns
        -------
        description : str
            The description of the object.
        Examples
        -------

        >>> # Describe the object in the image.
        >>> def execute_command(object_name, feed_back_image) -> str:
        >>>     giraffe = IsometricImage(object_name="Giraffe", feed_back_image=None)
        >>>     description = giraffe.describe_object(iter=1)
        >>>     return description
        """

        return self.describe_object(iter)
    
    def make_plan(self, description: str, iter: int)->str:
        """
        Make a plan to assemble the object.
        Parameters
        ----------
        description : str
            The description of the object.
        iter : int
            The iteration of the plan.
        Returns
        -------
        plan : str
            The plan of the object.
        Examples
        -------

        >>> # Make a plan to assemble the object.
        >>> def execute_command(object_name, feed_back_image) -> str:
        >>>     giraffe = IsometricImage(object_name="Giraffe", feed_back_image=None)
        >>>     description = giraffe.describe_object(iter=1)
        >>>     plan = giraffe.make_plan(description, iter=1)
        >>>     return plan
        """

        return self.make_plan(description, iter)

    def order_blocks(self, plan: str, iter: int)->str:
        """
        Make a list of blocks to assemble the object.
        Parameters
        ----------
        plan : str
            The plan of the object.
        iter : int
            The iteration of the order.
        Returns
        -------
        order : str
            The order of the object.
        Examples
        -------

        >>> # Make a list of blocks to assemble the object.
        >>> def execute_command(object_name, feed_back_image) -> str:
        >>>     giraffe = IsometricImage(object_name="Giraffe", feed_back_image=None)
        >>>     description = giraffe.describe_object(iter=1)
        >>>     plan = giraffe.make_plan(description, iter=1)
        >>>     order = giraffe.order_blocks(plan, iter=1)
        >>>     return order
        """
        return self.order_blocks(plan, iter)
    
    def decide_position(self, order: str, iter: int)->list[dict]:
        """
        Decide the position of the blocks to assemble the object.
        Parameters
        ----------
        order : str
            The order of the object.
        iter : int
            The iteration of the position.
        Returns
        -------
        positions : list[dict]
            The positions of the blocks.
        Examples
        -------

        >>> # Decide the position of the blocks to assemble the object.
        >>> def execute_command(object_name, feed_back_image) -> str:
        >>>     giraffe = IsometricImage(object_name="Giraffe", feed_back_image=None)
        >>>     description = giraffe.describe_object(iter=1)
        >>>     plan = giraffe.make_plan(description, iter=1)
        >>>     order = giraffe.order_blocks(plan, iter=1)
        """
        return self.decide_position(order, iter)
    
    def refine_structure(self, image: Image.Image, iter: int)->list[dict]:
        """
        Refine the structure to make it stable.
        Parameters
        ----------
        image : Image.Image
            The image of the structure.
        iter : int
            The iteration of the refinement.
        Returns
        -------
        positions : list[dict]
            The positions of the blocks.
        Examples
        -------

        >>> # Refine the structure to make it stable.
        >>> def execute_command(object_name, feed_back_image) -> str:
        >>>     giraffe = IsometricImage(object_name="Giraffe", feed_back_image=None)
        >>>     image = giraffe.get_structure_image()
        >>>     positions = giraffe.refine_structure(image, iter=1)
        >>>     return positions
        """
        return self.refine_structure(image, iter)
    
    def make_structure(self, positions: list[dict])->None:
        """
        Make the structure of the object in simulation follow the order and position.
        Parameters
        ----------
        positions : list[dict]
            The positions of the blocks.
        Examples
        -------

        >>> # Make the structure of the object in simulation follow the order and position.
        >>> def execute_command(object_name, feed_back_image) -> str:
        >>>     giraffe = IsometricImage(object_name="Giraffe", feed_back_image=None)
        >>>     positions = giraffe.decide_position(order, iter=1)
        >>>     giraffe.make_structure(positions)
        """
        return self.make_structure(positions)
    
    def get_structure_image(self)->Image.Image:
        """
        Get the image of the structure from three angles.
        Returns
        -------
        img: Image.Image
            The image of the structure.
        Examples
        -------

        >>> # Get the isometric image of the structure.
        >>> def execute_command(object_name, feed_back_image) -> str:
        >>>     giraffe = IsometricImage(object_name="Giraffe", feed_back_image=None)
        >>>     image = giraffe.get_structure_image()
        >>>     return image
        """
        return self.get_structure_image()
    def stability_check(self, blocks: list[dict])->list[bool, dict, dict, Image.Image, Image.Image]:
        """
        Check the stability of the structure.
        Parameters
        ----------
        blocks : list[dict]
            The blocks of the structure.
        Returns
        -------
        bool, dict, dict, Image.Image, Image.Image
            Whether the structure is stable, the unstable block, the position delta, the x image, the y image.
        Examples
        -------

        >>> # Check the stability of the structure.
        >>> def execute_command(object_name, feed_back_image) -> str:
        >>>     giraffe = IsometricImage(object_name="Giraffe", feed_back_image=None)
        >>>     blocks = giraffe.decide_position(order="giraffe", iter=1)
        >>>     is_stable, unstable_block, pos_delta, x_img, y_img = giraffe.stability_check(blocks)
        >>>     return is_stable, unstable_block, pos_delta, x_img, y_img
        """
        return self.stability_check(blocks)
    
    def save_structure(self) -> None:
        """
        Save the structure of the object in simulation.
        Parameters
        ----------
        Examples
        -------

        >>> # Save the structure of the object in simulation.
        >>> def execute_command(object_name, feed_back_image) -> str:
        >>>     giraffe = IsometricImage(object_name="Giraffe", feed_back_image=None)
        >>>     giraffe.save_structure()
        """
        return self.save_structure()
    
    def get_structure_rating(self, iter: int = 0) -> str: # Hoặc dict nếu hàm trả về dict đã parse
        """
        Get the rating of the structure.
        The returned string typically contains a JSON object with rating and reasoning.
        Parameters
        ----------
        iter : int
            The iteration number for caching or versioning.
        Returns
        -------
        str
            A string containing the rating information, usually as a JSON object.
        Examples
        -------
        >>> # Get the rating of the current structure
        >>> def execute_command(object_name, feed_back_image) -> str:
        >>>     structure_obj = IsometricImage(object_name="MyObject", feed_back_image=None)
        >>>     structure_
        >>>     # Assume structure is built and image is available internally or generated
        >>>     rating_info_str = structure_obj.get_structure_rating(iter=1)
        >>>     return rating_info_str
        """
        return self.get_structure_rating(iter)

    def get_structure_info(self, iter: int = 0) -> str: # Hoặc dict nếu hàm trả về dict đã parse
        """
        Get detailed information and resemblance guesses for the structure.
        The returned string typically contains a JSON object with guesses and description.
        Parameters
        ----------
        iter : int
            The iteration number for caching or versioning.
        Returns
        -------
        str
            A string containing structure information, usually as a JSON object.
        Examples
        -------
        >>> # Get information about the current structure
        >>> def execute_command(object_name, feed_back_image) -> str:
        >>>     structure_obj = IsometricImage(object_name="MyObject", feed_back_image=None)
        >>>     # Assume structure is built and image is available internally or generated
        >>>     info_str = structure_obj.get_structure_info(iter=1)
        >>>     return info_str
        """
        return self.get_structure_info(iter)
    
    
Write a function using Python and the Structure class (above) that could be executed to provide an answer to the query. 


### Examples
{example}

Plan at this step: {plan}
** Expected format output begin with **
def execute_command(object_name, feed_back_image):
'''

EXAMPLES_CODER = '''
### Example 1 
Plan:
Step 1: Get the general description of the table.
Step 2: Plan which blocks to use to represent the table.
A: ```
def execute_command(object_name, feed_back_image):
    # Initialize the structure
    table = IsometricImage(object_name, feed_back_image)
    # Step 1: Get the general description of the table.
    description = table.describe_object(iter=0)
    # Step 2: Plan which blocks to use to represent the table.
    plan = table.make_plan(description, iter=0)
    return plan
```

### Example 2
Plan:
Step 1: Decide the position of the blocks to assemble the table.
Step 2: Return the blocks position.
A: ```
def execute_command(object_name, feed_back_image):
    # Initialize the structure
    table = IsometricImage(object_name, feed_back_image)
    # Get description and plan first
    description = table.describe_object(iter=0)
    plan = table.make_plan(description, iter=0)
    # Step 1: Decide the position of the blocks to assemble the table.
    order = table.order_blocks(plan, iter=0)
    positions = table.decide_position(order, iter=0)
    # Step 2: Return the blocks position.
    return positions
```

### Example 3
Plan:
Step 1: Place the blocks in the simulation follow the order and position.
Step 2: Get the isometric image of the structure.
Step 3: Refine the design to make it stable.
Step 4: Save and return the structure.
A: ```
def execute_command(object_name, feed_back_image):
    # Initialize structure
    table = IsometricImage(object_name, feed_back_image)
    # Get description, plan and positions
    description = table.describe_object(iter=0)
    plan = table.make_plan(description, iter=0)
    order = table.order_blocks(plan, iter=0)
    positions = table.decide_position(order, iter=0)
    # Step 1: Place the blocks in the simulation follow the order and position.
    table.make_structure(positions)
    # Step 2: Get the isometric image of the structure.
    img = table.get_structure_image()
    # Step 3: Refine the design to make it stable.
    table.refine_structure(table.blocks)
    # Step 4: Save and return the structure.
    table.save_structure()
    return positions
```

### Example 4
Plan:
Step 1: Get the general description of the letter U.
Step 2: Plan which blocks to use to represent the letter U.
Step 3: Decide the position of the blocks to assemble the letter U.
Step 4: Return the blocks position.
A: ```
def execute_command(object_name, feed_back_image):
    # Initialize the structure
    letter = IsometricImage(object_name, feed_back_image)
    # Step 1: Get the general description of the letter U.
    description = letter.describe_object(iter=0)
    # Step 2: Plan which blocks to use to represent the letter U.
    plan = letter.make_plan(description, iter=0)
    # Step 3: Decide the position of the blocks to assemble the letter U.
    order = letter.order_blocks(plan, iter=0)
    positions = letter.decide_position(order, iter=0)
    # Step 4: Return the blocks position.
    return positions
```

### Example 5
Plan:
Step 1: Place the blocks in the simulation follow the order and position.
Step 2: Get the isometric image of the structure.
Step 3: Check stability of the structure.
Step 4: Save the structure and return success status.
A: ```
def execute_command(object_name, feed_back_image):
    # Initialize the structure
    letter = IsometricImage(object_name, feed_back_image)
    # Complete workflow to get positions
    description = letter.describe_object(iter=0)
    plan = letter.make_plan(description, iter=0)
    order = letter.order_blocks(plan, iter=0)
    positions = letter.decide_position(order, iter=0)
    # Step 1: Place the blocks in the simulation follow the order and position.
    letter.make_structure(positions)
    # Step 2: Get the isometric image of the structure.
    img = letter.get_structure_image()
    # Step 3: Check stability of the structure.
    is_stable, unstable_block, pos_delta, x_img, y_img = letter.stability_check(letter.blocks)
    # Step 4: Save the structure and return success status.
    letter.save_structure()
    return {"positions": positions, "is_stable": is_stable}
```

### Example 6
Plan:
Step 1: Build complete structure with refinement.
Step 2: Get structure rating.
Step 3: Get structure info and guesses.
Step 4: Return complete analysis.
A: ```
def execute_command(object_name, feed_back_image):
    # Initialize the structure
    structure = IsometricImage(object_name, feed_back_image)
    # Step 1: Build complete structure with refinement
    description = structure.describe_object(iter=0)
    plan = structure.make_plan(description, iter=0)
    order = structure.order_blocks(plan, iter=0)
    positions = structure.decide_position(order, iter=0)
    structure.make_structure(positions)
    img = structure.get_structure_image()
    structure.refine_structure(structure.blocks)
    # Step 2: Get structure rating
    rating_info = structure.get_structure_rating(iter=0)
    # Step 3: Get structure info and guesses
    structure_info = structure.get_structure_info(iter=0)
    # Step 4: Return complete analysis
    structure.save_structure()
    return {
        "positions": positions,
        "rating": rating_info,
        "info": structure_info
    }
```
'''