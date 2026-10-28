from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
import os
load_dotenv()
# Initialize the OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)


# Define the function to generate spoken audio with parameters
def generate_user_audio(text="This is a test of the text-to-speech functionality.", voice="shimmer"):
    # Define the file name for the output audio
    file_name = "test.mp3"

    # Path to save the audio file
    speech_file_path = Path(__file__).parent / file_name

    # Generate the spoken audio
    response = client.audio.speech.create(
        model='tts-1',
        voice=voice,
        input=text
    )

    # Save the audio file
    with open(speech_file_path, 'wb') as f:
        for chunk in response.iter_bytes():
            f.write(chunk)

    print(f"Audio saved to {speech_file_path}")


# Example usage
if __name__ == "__main__":
    # Example inputs
    example_voice = "fable"
    example_text = f"Hello, this is {example_voice}. How are you doing today? Are you having a good day so far?"

    # Generate audio with example inputs
    generate_user_audio(example_text, example_voice)