from utils import markdown_json

def get_info_prompt() -> list:
    """
    Get the prompt for getting structure information.
    
    Returns:
        List containing the prompt elements
    """
    return [
        f"""
I am currently building a structure made out of toy blocks shown in the given image. Describe in detail as much as you can about this image. Please list the top 10 things that the structure most resembles in order of similarity.

After describing the image in detail and providing some initial thoughts, answer as JSON in the following format providing 10 guesses. Your guesses should mostly be single words like "bottle" and never use adjectives like "toy_bottle". 
{
    markdown_json({"guesses": ["guess_1", "guess_2", "guess_3", "guess_4", "guess_5", "guess_6", "guess_7", "guess_8", "guess_9", "guess_10"]})
}
""".strip()
    ]
