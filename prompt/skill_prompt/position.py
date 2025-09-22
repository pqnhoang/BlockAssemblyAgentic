from utils import markdown_json

def get_position_prompt(object_name: str, order: str) -> str:
    """
    Get the prompt for deciding positions of blocks.
    
    Args:
        object_name: The name of the object to assemble
        order: The stacking order of blocks
        
    Returns:
        The formatted prompt string
    """
    return f"""
With the stacking order determined as {order}, I need to know the precise x, y, z positions and yaw, pitch, roll orientation (in degrees) for each block to build a '{object_name}' structure.

General Rules:
1.  The structure should be stable and centered around the origin (0, 0).
2.  Blocks at similar heights must be spaced out in x and y to avoid collision.
3.  Every block must stack correctly on its specified base (or the ground).
4.  The `yaw` angle is the rotation around the vertical z-axis.

Block-Specific Instructions:

**For Rigid Blocks (cuboid, cylinder):**
-   The `x`, `y` coordinates represent the center of the block.
-  `pitch` and `roll` angles are always 0.
-   For rigid blocks, the "baseblock" is always "None" since they are not connected to any other block.

**For Joint Blocks:**
-   You must specify the `base_block` that the joint is connected to.
-   The `x`, `y`, `z` in the 'offset' represent the starting position of the joint's base, relative to the center of its `base_block`.
-   The `yaw`, `pitch`, and `roll` angles determine the initial orientation of the joint's base. Think of it like attaching a robotic arm to a body.
    -   **Attaching on TOP:** If a joint is placed on the top surface of a horizontal block (like a snake on the floor), its `pitch` and `roll` will typically be 0. The `yaw` will control its direction.
    -   **Attaching to the SIDE:** If a joint needs to stick out horizontally from the side of a block (like an arm on a torso), you MUST use `pitch` or `roll`.
        -   **Example:** A `pitch` of 90 degrees will make the joint point straight out, perpendicular to the vertical side of the `base_block`.
        -   **Example:** A `roll` can then be used to twist the joint's bending axis. A `roll` of 90 degrees would change its bending motion from up-down to left-right.
- Careful design the position x, y, z. **Crucial Calculation Example:** To attach a block to the **edge** of a cylinder `base_block` that has a `radius` of 20 and is centered at its own (0,0), the attached block's `offset` must have `x: 20` (or `y: 20`) to be perfectly on the edge. An `x` value like 25 would be incorrect and floating in space.
Provide your reasoning for the chosen positions and orientations for each block.

Output a JSON following this exact format:
{markdown_json(
    [
        {
            "name": "support1",
            "shape": "cylinder",
            "dimensions": {"radius": 20, "height": 40},
            "color": [0.5, 0.5, 0.5, 1],
            "offset": {"x": -50, "y": 0, "z": 0, "yaw": 0, "pitch": 0, "roll": 0},
            "base_block": "None"
        },
        {
            "name": "deck",
            "shape": "cuboid",
            "dimensions": {"x": 100, "y": 50, "z": 20},
            "color": [0.5, 0.5, 0.5, 1],
            "offset": {"x": 0, "y": 0, "z": 40, "yaw": 0, "pitch": 0, "roll": 0},
            "base_block": "support1",
        },
        {
            "name": "tentacle_arm",
            "shape": "joint",
            "dimensions": {"x": 100, "y": 10, "z": 10},
            "color": [0.5, 0.5, 0.5, 1],
            "offset": {"x": 0, "y": 30, "z": 10, "yaw": 0, "pitch": 90, "roll": 0},
            "base_block": "deck",
        }
    ]
)}
"""
