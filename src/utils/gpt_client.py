from io import BytesIO
from openai import OpenAI
from PIL import Image
import numpy as np
import base64
from dotenv import load_dotenv
import requests

load_dotenv()

def encode_image_from_pil(pil_img: Image.Image):
    # convert to RGB
    pil_img = pil_img.convert("RGB")

    buffered = BytesIO()
    pil_img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def encode_image_from_array(image: np.ndarray):
    # convert to PIL
    pil_img = Image.fromarray(image)
    # encode to base64
    return encode_image_from_pil(pil_img)


def encode_image_from_file(image_path):
    # open img in PIL
    with Image.open(image_path) as img:
        # encode to base64
        return encode_image_from_pil(img)


def download_image(image_url, file_path):
    # Download the image
    response = requests.get(image_url)
    if response.status_code == 200:
        with open(file_path, "wb") as file:
            file.write(response.content)
        print(f"Image successfully downloaded to {file_path}")
        return True
    else:
        print(f"Failed to download image. Status code: {response.status_code}")
        return False


class GPTClient:
    _instance = None

    @classmethod
    def _get_instance(cls):
        if cls._instance is None:
            cls._instance = OpenAI()  # Initialize the client once
        return cls._instance

    @classmethod
    def encode_messages_and_images(cls, messages_and_images):
        def _encode_image_content(img):
            if type(img) == str:
                base64_image = encode_image_from_file(img)
            elif type(img) == np.ndarray:
                base64_image = encode_image_from_array(img)
            else:
                base64_image = encode_image_from_pil(img)

            return {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                },
            }

        def _encode_text_content(message):
            return {"type": "text", "text": message}

        content = []
        for item in messages_and_images:
            if (
                type(item) == str
            ):  # and not os.path.exists(item): if the img input is as a file path this won't work :(
                content.append(_encode_text_content(item))
            else:
                content.append(_encode_image_content(item))

        return content

    @classmethod
    def prompt_gpt(
        cls,
        messages_and_images,
        context=[],
        max_tokens=3500,
        temperature=0,
        system_message=None,
    ):
        """
        get full gpt response object
        """
        client = cls._get_instance()

        content = cls.encode_messages_and_images(messages_and_images)

        if system_message:
            messages = [
                {"role": "system", "content": system_message},
                *context,
                {"role": "user", "content": content},
            ]
        else:
            messages = [*context, {"role": "user", "content": content}]

        # call gpt
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        # get message from chat completion
        response_message = response.choices[0].message.model_dump()

        # get text response
        response_txt = response_message["content"]
        # add response to context
        new_context = messages + [response_message]

        return response_txt, new_context

    @classmethod
    def get_image_from_dalle(cls, prompt, img_save_path):
        client = cls._get_instance()
        # Generate the image using DALL-E
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            n=1,
        )
        # Extract the image URL from the response
        image_url = response.data[0].url

        # Download the image
        ok = download_image(image_url, img_save_path)

        # load image
        if ok:
            with Image.open(img_save_path) as img:
                img = img.convert("RGB")
                return img
        else:
            raise Exception("Failed to download DALLE image")