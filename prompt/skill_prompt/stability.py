from utils import markdown_json
from structure import Block

def get_stability_prompt(to_build: str, unstable_block: Block, pos_delta: list, structure_json: dict, x_img, y_img) -> list:
    """
    Get the prompt for stability correction.
    
    Args:
        to_build: The name of the object being built
        unstable_block: The unstable block
        pos_delta: Position change of the unstable block
        structure_json: Current structure JSON
        x_img: Side view image (x-axis)
        y_img: Side view image (y-axis)
        
    Returns:
        List of prompt elements including text and images
    """
    return [
        f"""
{markdown_json(structure_json)}

While building the {to_build} by placing blocks one at a time in the order you specified by the JSON above, I noticed that block {unstable_block.gpt_name} is unstable and falls. 
The block moved by {pos_delta[0]:.2f} mm in the x direction and {pos_delta[1]:.2f} mm in the y direction.
Please adjust the position of block {unstable_block.gpt_name} (And potentially other blocks) to make the structure more stable.
Make sure every block has a stable base to rest on.

Output the JSON with your corrections following the same format and provide some reasoning for the changes you made. Feel free to correct other parts of the structure if they appear incorrect or to add, change, or remove blocks.

Here is an orthographic image of the side view of the structure with the y-axis pointing to the right and the z-axis pointing up. {unstable_block.gpt_name} is highlighted in red while the other blocks are colored in white.
""",
        x_img,
        f"""
Here is an orthographic image of the side view of the structure with the x-axis pointing to the right and the z-axis pointing up. {unstable_block.gpt_name} is highlighted in red while the other blocks are colored in white.
""",
        y_img,
        """
Describe what you see in these images and use them to help inform your correction. Then, provide the ouptut JSON in the proper format.
"""
    ]
