def get_order_prompt(object_name: str, plan: str) -> str:
    """
    Get the prompt for ordering blocks in a plan.
    
    Args:
        object_name: The name of the object to assemble
        plan: The assembly plan
        
    Returns:
        The formatted prompt string
    """
    return f"""
Given the blocks described in the plan {plan}, I will place and stack these blocks one at a time by lowering them from a very tall height.

Please describe the sequence in which the blocks should be placed to correctly form a {object_name} structure. This means that blocks at the bottom should be placed first, followed by the higher blocks, so that the upper blocks can be stacked on top of the lower ones. Also note that it is difficult to stack blocks on top of a cone, so avoid placing blocks directly on top of cones.

For each block, specify whether it will be placed on the ground or on top of another block. If a block will be supported by multiple blocks, mention all of them. Ensure that the blocks are placed in a way that they remain stable and won't topple over when the physics simulation is run. Blocks cannot hover without support.
""".strip()
