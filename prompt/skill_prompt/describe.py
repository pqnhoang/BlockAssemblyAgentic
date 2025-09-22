def get_describe_prompt(object_name: str) -> str:
    """
    Get the prompt for describing an object.
    
    Args:
        object_name: The name of the object to describe
        
    Returns:
        The formatted prompt string
    """
    return f"""
I'm working on constructing a block tower that represents a(n) {object_name}. 
Following the image instruction, provide a qualitative description of the design that captures its essence in a minimalistic style. The design should focus on simplicity, avoiding unnecessary complexity while still conveying the key features of a(n) {object_name}. The description should highlight the overall structure and proportions, emphasizing how the block arrangement reflects the object's shape and form. However the design shouldn't be too large, too wide, or too tall. The available blocks are not suitable to demonstrate the details of the object (e.g eyes, ears, ...), so you should focus on the overall structure and proportions.
""".strip()
