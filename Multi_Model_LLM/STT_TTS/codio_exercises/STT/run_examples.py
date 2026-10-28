"""Inspect and transcribe a real bundled WAV file."""

import wave
import os
from pathlib import Path
from dotenv import load_dotenv
from openai import APIConnectionError, InternalServerError, OpenAI

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
MODEL = "whisper-1"
AUDIO_PATH = Path("media/box-chocs.wav")
client = OpenAI(api_key=api_key)


def inspect_wav(path: Path) -> tuple[int, int, float]:
    """Return sample rate, channel count and duration for a real WAV file."""
    with wave.open(str(path), "rb") as audio:
        sample_rate = audio.getframerate()
        channels = audio.getnchannels()
        duration = audio.getnframes() / sample_rate
    return sample_rate, channels, duration


def compact(value: object, limit: int = 140) -> str:
    """Collapse transcription output to one short terminal line."""
    return " ".join(str(value).split())[:limit]


def main() -> None:
    """Show real audio metadata followed by a live transcription."""
    sample_rate, channels, duration = inspect_wav(AUDIO_PATH)

    print(f"File: {AUDIO_PATH.name}")
    print(
        f"Audio: {sample_rate} Hz, "
        f"{channels} channel(s), "
        f"{duration:.1f} seconds"
    )
    print(f"Model: {MODEL}")

    try:
        with AUDIO_PATH.open("rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model=MODEL,
                file=audio_file,
                response_format="text",
            )
    except (APIConnectionError, InternalServerError) as error:
        print(
            "Transcription could not be completed because "
            "the API is temporarily unavailable."
        )
        print(
            "Please wait briefly, then run: "
            "python3 run_examples.py"
        )
        print(f"Error type: {type(error).__name__}")
        return

    print(f"Transcript: {compact(transcript)}")


if __name__ == "__main__":
    main()
