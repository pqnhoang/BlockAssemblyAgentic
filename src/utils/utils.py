import importlib.util
import os
import sys
import re
import base64
from PIL import Image
import io
import unicodedata
import re
import numpy as np
import json
import os
from io import BytesIO


def write_error(file_path, error_text):
    with open(file_path, "w") as file:
        file.write(error_text)


def switch_file_extension(file_path, new_extension):
    directory = os.path.dirname(file_path)
    filename, _ = os.path.splitext(os.path.basename(file_path))
    return os.path.join(directory, f"{filename}.{new_extension}")


def add_suffix_to_filename(path, suffix):
    directory = os.path.dirname(path)
    filename, extension = os.path.splitext(os.path.basename(path))
    new_filename = f"{filename}{suffix}{extension}"
    return os.path.join(directory, new_filename)


def slugify(value, allow_unicode=False):
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize("NFKC", value)
    else:
        value = (
            unicodedata.normalize("NFKD", value)
            .encode("ascii", "ignore")
            .decode("ascii")
        )
    value = re.sub(r"[^\w\s-]", "", value.lower())
    return re.sub(r"[-\s]+", "-", value).strip("-_")


def save_base64_image(base64_str, file_path):
    # Extract base64 image data
    base64_data = base64_str.split(",")[1]
    image_data = base64.b64decode(base64_data)
    # Save image
    image = Image.open(BytesIO(image_data))
    image.save(file_path)


# Function to convert NumPy arrays to lists
def convert_numpy_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def save_to_json(data, file_path):
    # Tạo thư mục nếu chưa tồn tại
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"Created directory: {directory}")
        except Exception as e:
            print(f"Error creating directory {directory}: {e}")
            raise
    
    # Tiến hành lưu file
    try:
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(data, file, default=convert_numpy_to_list, indent=4)
        print(f"Successfully saved data to {file_path}")
    except Exception as e:
        print(f"Error saving to {file_path}: {e}")
        raise


def markdown_json(data):
    pretty_data = json.dumps(data, default=convert_numpy_to_list)
    return f"```json\n{pretty_data}\n```"


def load_from_json(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_file(data, file_path):
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(data)


def load_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def import_function_from_file(file_path, function_name):
    # Create a module spec from the file location
    spec = importlib.util.spec_from_file_location("module.name", file_path)

    # Create a module from the spec
    module = importlib.util.module_from_spec(spec)

    # Add the module to sys.modules
    sys.modules["module.name"] = module

    # Execute the module
    spec.loader.exec_module(module)

    # Get the function
    function = getattr(module, function_name)

    return function


def extract_code_from_response(
    gpt_output: str, lang: str = "python", last_block_only=False
) -> str:
    # Regular expression to match Python code enclosed in '''python ... '''
    code_blocks = re.findall(rf"```{lang}(.*?)```", gpt_output, re.DOTALL)

    if last_block_only:
        return code_blocks[-1]
    # Concatenate all code blocks into a single string
    concatenated_code = "\n".join(code_block.strip() for code_block in code_blocks)

    return concatenated_code


def get_last_json_as_dict(gpt_output):
    try:
        return json.loads(
            extract_code_from_response(gpt_output, lang="json", last_block_only=True)
        )
    except:
        print(gpt_output)
        print("ERROR in code extraction")
        raise AssertionError("Terminating Process Early Because of bad JSON Response")

def print_dict(dict_data):
    """
    Print information about furniture components from a dictionary list
    in a structured and readable format.
    
    Args:
        furniture_list: List of dictionaries containing furniture component information
    """
    if not dict_data:
        print("Empty list!")
        return
        
    for idx, item in enumerate(dict_data, 1):
        print(f"\n{'='*50}")
        print(f"COMPONENT {idx}: {item.get('name', 'No name')}")
        print(f"{'='*50}")
        
        print(f"- Shape: {item.get('shape', 'N/A')}")
        
        # Print dimensions
        dimensions = item.get('dimensions', {})
        print("- Dimensions:")
        for dim_name, dim_value in dimensions.items():
            print(f"  + {dim_name}: {dim_value} m")
        
        # Print color
        color = item.get('color', [])
        if len(color) == 4:
            print(f"- Color (RGBA): R={color[0]}, G={color[1]}, B={color[2]}, A={color[3]}")
        else:
            print(f"- Color: {color}")
        
        # Print position
        position = item.get('position', {})
        print("- Position:")
        for pos_axis, pos_value in position.items():
            print(f"  + {pos_axis}: {pos_value} m")
        
        # Print rotation
        print(f"- Rotation (yaw): {item.get('yaw', 'N/A')} rad")
        
    print("\n" + "-"*50)
    print(f"Total components: {len(dict_data)}")
def visualize_isometric(img: Image.Image):
    """
    Visualize the grasp pose on the image.

    Parameters:
    - image: The input image as a numpy array.
    - grasp_pose: A tuple (quality, x, y, w, h, angle) representing the grasp pose.
        - x, y: Center of the rectangle.
        - w, h: Width and height of the rectangle.
        - angle: Rotation angle of the rectangle in degrees.

    Returns:
    """
    point_color1 = (255, 255, 0)  # BGR  
    point_color2 = (255, 0, 255)  # BGR
    thickness = 2
    lineType = 4
    x, y, w, h, angle = grasp_pose[1:]
    center = (x, y)
    size = (w, h)
    box = cv2.boxPoints((center, size, angle))
    box = np.int64(box)
    plt.figure(figsize=(5,5))
    cv2.line(image, box[0], box[3], point_color1, thickness, lineType)
    cv2.line(image, box[3], box[2], point_color2, thickness, lineType)
    cv2.line(image, box[2], box[1], point_color1, thickness, lineType)
    cv2.line(image, box[1], box[0], point_color2, thickness, lineType)
    plt.imshow(image)
    plt.axis('off')
    output_path = '/LOCAL2/anguyen/faic/quang/viper_duality/imgs/grasp_pose_visualization.png'
    plt.savefig(output_path)
    return Path(output_path)