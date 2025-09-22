from utils import markdown_json

def get_rating_prompt(object_name: str) -> str:
    """
    Get the prompt for rating a structure.
    
    Args:
        object_name: The name of the object being rated
        
    Returns:
        The formatted prompt string
    """
    return f"""
Given your description of the block structure, how well does the structure in the image use blocks to resemble a {object_name} considering that the structure is made from  a limited set of toy blocks? Rate the resemblance of the block structure to a {object_name} on a scale of 1 to 5 defined by the following:
    1 - the structure in the image has no resemblance to the intended structure. It's missing all key features and appears incoherent
    2 - the structure in the image has a small amount of resemblance to the intented structure. It has at least 1 key feature and shows an attempt at the intended structure
    3 - the structure has clear similarities to the intended structure and appears coherent. It has at least 1 key feature and shows similarities in other ways as well.
    4 - the structure represents multiple key features of the intended design and shows a decent overall resemblance.
    5 - the structure in the image is a good block reconstruction of the intended structure, representing multiple key features and showing an overall resemblance to the intended structure.

Provide a brief explanation of your though process then provide your final response as JSON in the following format:
{markdown_json({"rating": 5})}
"""
