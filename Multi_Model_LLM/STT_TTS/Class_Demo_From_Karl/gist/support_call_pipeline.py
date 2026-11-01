"""
Support call pipeline demo.

Pipeline steps:
  1. Setup & verification    - check env vars, ffmpeg, input file
  2. Audio preprocessing     - convert to 16kHz mono WAV
  3. Transcription (ASR)     - speech-to-text via OpenAI
  4. Retrieval (embeddings)  - find similar past calls
  5. Response generation     - LLM answer grounded in transcript + retrieval
  6. Text-to-speech (TTS)    - spoken output of the answer
"""

from pathlib import Path
import argparse
import os
import re
import shutil
import subprocess

import numpy as np
import pandas as pd
import librosa
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from IPython.display import Audio, display
from openai import OpenAI


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

# Paths are anchored to this file, so the script works from any directory.
#   media/inputs/                          - put your recording here
#   media/outputs/support_call_pipeline/   - generated files (safe to delete)
MEDIA_DIR = Path(__file__).resolve().parent / "media"
OUTPUT_DIR = MEDIA_DIR / "outputs" / Path(__file__).stem

INPUT_AUDIO = MEDIA_DIR / "inputs" / "sample_support_call.m4a"
CLEAN_AUDIO = OUTPUT_DIR / "clean_audio_16k_mono.wav"
TTS_OUTPUT = OUTPUT_DIR / "answer_openai.wav"

STT_MODEL = os.getenv("OPENAI_STT_MODEL", "gpt-4o-transcribe")
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
ANSWER_MODEL = os.getenv("OPENAI_ANSWER_MODEL", "gpt-5-mini")
TTS_MODEL = os.getenv("OPENAI_TTS_MODEL", "tts-1")
TTS_VOICE = os.getenv("OPENAI_TTS_VOICE", "coral")

FALLBACK_TRANSCRIPT = (
    "Hi, I cannot log into my account. "
    "The password reset email never arrives, and I need access today."
)

PAST_CALLS = [
    {
        "id": "call_101",
        "topic": "billing",
        "transcript": "The customer was charged twice for the same invoice and wants a refund.",
    },
    {
        "id": "call_102",
        "topic": "password_reset",
        "transcript": "The user cannot log in because the password reset email is not arriving.",
    },
    {
        "id": "call_103",
        "topic": "delivery_delay",
        "transcript": "The customer asks why their package is delayed and wants a new delivery date.",
    },
    {
        "id": "call_104",
        "topic": "audio_issue",
        "transcript": "The microphone is not detected during video calls after a laptop update.",
    },
]


# --------------------------------------------------------------------------
# Console: step banners with an optional pause
# --------------------------------------------------------------------------

BANNER_WIDTH = min(shutil.get_terminal_size(fallback=(72, 24)).columns, 72)

BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
RESET = "\033[0m"


def step_banner(label: str, pause: bool = True) -> None:
    """Print a separator block for the next step and, in live demo mode,
    wait for Enter so the presenter can narrate one step at a time."""
    rule = "━" * BANNER_WIDTH
    print(f"\n{CYAN}{rule}{RESET}")
    print(f"{BOLD}  ▶ {label}{RESET}")
    print(f"{CYAN}{rule}{RESET}")
    if pause:
        try:
            input(f"{DIM}  Press Enter to continue...{RESET}")
        except EOFError:
            # Non-interactive run (piped input) - carry on without pausing.
            pass
        print()


# --------------------------------------------------------------------------
# Step 1: Setup and verification
# --------------------------------------------------------------------------

def verify_environment(client: OpenAI) -> None:
    """Check that required env vars, tooling, and input files are present."""
    print("OpenAI key set:", bool(os.environ.get("OPENAI_API_KEY")))
    print("FFmpeg available:", bool(shutil.which("ffmpeg")))
    print("Input audio exists:", INPUT_AUDIO.exists())
    print("Answer model:", ANSWER_MODEL)


# --------------------------------------------------------------------------
# Step 2: Audio preprocessing
# --------------------------------------------------------------------------

def preprocess_audio(input_path: Path, output_path: Path) -> None:
    """Convert input audio to 16kHz mono WAV using ffmpeg, then report stats."""
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(input_path),
        "-ar", "16000",
        "-ac", "1",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    print("Saved", output_path, "with FFmpeg")

    audio, sample_rate = librosa.load(output_path, sr=16000, mono=True)
    duration = len(audio) / sample_rate

    print("Sample rate:", sample_rate)
    print("Duration:", round(duration, 2), "seconds")
    print("Samples:", audio.shape)

    display(Audio(str(output_path), autoplay=False))


# --------------------------------------------------------------------------
# Step 3: Transcription (ASR)
# --------------------------------------------------------------------------

def transcribe_audio(client: OpenAI, audio_path: Path) -> str:
    """Transcribe audio via OpenAI STT model."""
    with audio_path.open("rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model=STT_MODEL,
            file=audio_file,
            prompt="A short customer support call",
        )
    return transcription.text.strip()


def get_transcript(client: OpenAI) -> str:
    """Return a transcript: from audio if available, else the fallback text."""
    if not INPUT_AUDIO.exists():
        print("Missing", INPUT_AUDIO, "- using fallback transcript:")
        print(FALLBACK_TRANSCRIPT)
        return FALLBACK_TRANSCRIPT

    preprocess_audio(INPUT_AUDIO, CLEAN_AUDIO)
    transcript = transcribe_audio(client, CLEAN_AUDIO)
    print(transcript)
    return transcript


# --------------------------------------------------------------------------
# Step 4: Retrieval (embeddings)
# --------------------------------------------------------------------------

def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-12, None)


def embed_texts(client: OpenAI, texts: list[str]) -> np.ndarray:
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )
    vectors = np.array([item.embedding for item in response.data], dtype=np.float32)
    return normalize_rows(vectors)


def retrieve_similar_calls(
    client: OpenAI, transcript: str, past_calls: list[dict], top_k: int = 2
) -> list[dict]:
    """Embed the transcript and past calls, rank by cosine similarity."""
    past_texts = [item["transcript"] for item in past_calls]

    past_embeddings = embed_texts(client, past_texts)
    query_embedding = embed_texts(client, [transcript])

    scores = cosine_similarity(query_embedding, past_embeddings)[0]

    results = pd.DataFrame({
        "id": [item["id"] for item in past_calls],
        "topic": [item["topic"] for item in past_calls],
        "transcript": past_texts,
        "score": scores,
    }).sort_values("score", ascending=False)

    print(results[["id", "topic", "score"]].to_string(index=False))

    return results.head(top_k).to_dict(orient="records")


# --------------------------------------------------------------------------
# Step 5: Response generation (LLM)
# --------------------------------------------------------------------------

def generate_answer(client: OpenAI, transcript: str, retrieved_context: list[dict]) -> str:
    """Ask the LLM to summarize, classify intent, and suggest a next action."""
    user_prompt = (
        "New call transcript:\n"
        f"{transcript}\n\n"
        "Similar past calls:\n"
        f"{retrieved_context}\n\n"
        "Task:\n"
        "1. Summarize the user issue in one sentence.\n"
        "2. Identify the likely intent.\n"
        "3. Suggest the next best support action.\n"
        "Keep the answer short and clear."
    )

    response = client.responses.create(
        model=ANSWER_MODEL,
        input=[
            {
                "role": "developer",
                "content": (
                    "You are a concise support assistant. "
                    "Use only the transcript and retrieved context. "
                    "Do not invent account details, ticket IDs, or policies."
                ),
            },
            {"role": "user", "content": user_prompt},
        ],
    )

    answer = response.output_text.strip()
    print(answer)
    return answer


# --------------------------------------------------------------------------
# Step 6: Text-to-speech (TTS)
# --------------------------------------------------------------------------

def clean_for_tts(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9.,!?;:()'\" ]+", "", text)
    return text.strip()


def synthesize_speech(client: OpenAI, text: str, output_path: Path) -> None:
    """Convert text to spoken audio and save it to output_path."""
    spoken_answer = clean_for_tts(text)
    print(spoken_answer)

    with client.audio.speech.with_streaming_response.create(
        model=TTS_MODEL,
        voice=TTS_VOICE,
        input=spoken_answer,
        instructions="Speak clearly and calmly as a customer support assistant.",
        response_format="wav",
    ) as response:
        response.stream_to_file(output_path)

    print("Saved spoken answer to", output_path)
    display(Audio(str(output_path), autoplay=False))


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-pause",
        action="store_true",
        help="Run straight through without waiting for Enter between steps.",
    )
    args = parser.parse_args()
    pause = not args.no_pause

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    client = OpenAI()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    step_banner("Step 1 - Setup and verification: verify_environment()", pause)
    verify_environment(client)

    step_banner("Steps 2-3 - Audio preprocessing and transcription: get_transcript()", pause)
    transcript = get_transcript(client)

    step_banner("Step 4 - Retrieval (embeddings): retrieve_similar_calls()", pause)
    retrieved_context = retrieve_similar_calls(client, transcript, PAST_CALLS)

    step_banner("Step 5 - Response generation: generate_answer()", pause)
    answer = generate_answer(client, transcript, retrieved_context)

    step_banner("Step 6 - Text-to-speech: synthesize_speech()", pause)
    synthesize_speech(client, answer, TTS_OUTPUT)


if __name__ == "__main__":
    main()
