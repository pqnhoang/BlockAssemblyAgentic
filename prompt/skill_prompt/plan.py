from utils import markdown_json

def get_plan_prompt(object_name: str, description: str, available_blocks: dict) -> str:
    """
    Get the prompt for making a plan to assemble an object.
    
    Args:
        object_name: The name of the object to assemble
        description: Description of the object layout
        available_blocks: Dictionary of available blocks
        
    Returns:
        The formatted prompt string
    """
    return f"""
Here's a description of the layout of a {object_name}:
{description}

You have the following blocks available: 
{markdown_json(available_blocks)}
Write a plan for how to assemble a {object_name} using the available blocks. Use blocks as needed while respecting the number available constraint. 

Explain which blocks to use and their shape and dimensions. 

Explain the overall orientation of the structure.

Explain each block's role in the structure. 

Explain how the blocks should stack on top of each other (they can also just be placed on the ground). 

Do not overcomplicate the design. Try to use a minimal number of blocks to represent the key components of a {object_name}. Avoid making structures that are too tall, wide, or complex.

Only consider the main key components of a {object_name}, not minor details that are hard to represent with blocks. 
Use a minimal amount of blocks and keep it simple, just enough so that it looks like a {object_name}.

The dimensions of a cuboid are given as x, y, and z, which define the size of the block. You can rearrange these dimensions to fit your design requirements. For instance, if you need to place a block "vertically" with its longest side along the z-axis, but the dimensions are listed as x: 30, y: 30, z: 10, you can adjust them to x: 10, y: 30, z: 30 to achieve the desired orientation. Ensure the x and y dimensions are also consistent with the rest of the design.

Cylinders are always positioned "upright," with the flat sides parallel to the ground and their radius extending along the x and y axes.

Cones are always positioned with their flat side down and their pointed tip facing upwards. This means the base of the cone lies parallel to the ground plane, with the cone's height extending along the z-axis and the radius along the x and y axes.

For soft blocks (joint shape), these are flexible blocks that can bend and flex like soft fingers, tentacles, or flexible connectors. They comprise from multiple small block segments connected by joints, allowing them to deform naturally under forces. Use these when you need flexible parts (e.g tails of an animal, tentacles of octopus, flexible supports, ...). They have mass properties that affect stability, so account for their weight in the overall structure balance. Soft blocks can be good connectors between rigid parts or as dynamic elements that move. 

**Important**: To ensure the mass properties, we need to specify the base block that the soft block is connected to, and the joint angles between the segments. The base block is the first block in the soft block's list of blocks, and the joint angles are the angles between each segment in the soft block.

Decide a semantic name for the block for the role it represents in the structure. 
Decide the colors of each block to look like a {object_name}. Color is an rgba array with values from 0 to 1.
"""
