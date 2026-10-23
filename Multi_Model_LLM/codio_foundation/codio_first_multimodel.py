"""Send a real bundled image to the OpenAI Responses API."""

import base64
import mimetypes
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
"""load_dotenv() searches from the calling script’s directory upward until it finds a .env file. 
So both exercises above can find the file in my_project. Interactive sessions and debugging can use 
the current working directory as the starting point.
"""

MODEL = "gpt-5-mini"
IMAGE_PATH = Path("media/images/red-bicycle.png")
client = OpenAI()


def image_data_url(path: Path) -> str:
    """Encode a local image as a data URL accepted by the Responses API."""
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
        raise ValueError(f"Unsupported image type: {path.suffix}")
    media_type = mimetypes.guess_type(path.name)[0] or "image/png" #This line guesses the file’s MIME type from its filename, using "image/png" as a fallback.
    """
    Breaking it down:
    - path.name gets the filename, for example "photo.jpg".
    - mimetypes.guess_type(...) returns a tuple: (type, encoding), for example ("image/jpeg", None).
    - [0] takes the first item: "image/jpeg".
    - or "image/png" uses "image/png" if the guessed type is None.
    """

    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    # This line reads a file and converts its contents into Base64 text—useful when sending an image to an API.
    # why doing this? in this way it can be send over to the receiver in a JSON format, JSON only accepts text, ascii.
    """
    It works from the inside outward:
    1. path.read_bytes() reads the file’s contents as bytes.
    2. base64.b64encode(...) encodes those bytes into Base64, but returns the result as a Python bytes object.
    3. .decode("ascii") converts those Base64 bytes into a regular Python string (str).
    
    to decode:
    original_bytes = base64.b64decode(encoded)
    """
    return f"data:{media_type};base64,{encoded}"
    """ Here the returned media_type is needed for the receiving side to interpret back the received and decoded bytes
    to it's original format
    """

def compact(value: object, limit: int = 140) -> str:
    """Collapse model output to one short terminal line."""
    return " ".join(str(value).split())[:limit]


def main() -> None:
    """Analyse a real image and print only a short observable result."""
    response = client.responses.create(
        model=MODEL,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "Identify the following from the image:"
                            " 1. The main object"
                            " 2. its most dominant colour "
                            " 3. identify 1 visible feature that would affect night-time visibility"
                            " Do not use more than 20 words"
                        ),
                    },
                    {
                        "type": "input_image",
                        # the image is send over as a string, consisting of the media_type and the image encoded as text
                        "image_url": image_data_url(IMAGE_PATH),
                        "detail": "auto",
                    },
                ],
            }
        ],
    )
    print(f"Model: {MODEL}")
    print(f"Image: {IMAGE_PATH.name}")
    print(f"Analysis: {compact(response.output_text)}")


if __name__ == "__main__":
    main()
