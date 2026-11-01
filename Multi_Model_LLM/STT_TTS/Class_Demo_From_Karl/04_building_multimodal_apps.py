
"""
Multimodal video assistant demo.

Mirrors 04_building_multimodal_apps.ipynb step for step. Decomposes a
video into frames, audio, transcript segments, timestamped records,
embeddings, retrieval results, and a grounded answer.

Run from anywhere - paths come from media_paths.py.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from IPython.display import display, Image as IPImage, Video


from openai import OpenAI

from demo_ui import step_banner
from media_paths import SUPPORT_VIDEO, output_dir


load_dotenv(Path(__file__).resolve().parents[1] / ".env")

# Pass --no-pause to run straight through without waiting for Enter.
PAUSE = "--no-pause" not in sys.argv


# --------------------------------------------------------------------------
# Step 1 - Setup
# --------------------------------------------------------------------------

step_banner("Step 1 - Setup", PAUSE)

VIDEO_PATH = SUPPORT_VIDEO
WORK_DIR = output_dir(__file__)
FRAME_DIR = WORK_DIR / "frames"
AUDIO_PATH = WORK_DIR / "extracted_audio.wav"

STT_MODEL = os.getenv("OPENAI_STT_MODEL", "gpt-4o-transcribe")
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
USE_OPENAI_LLM = os.getenv("USE_OPENAI_LLM", "false").lower() == "true"
ANSWER_MODEL = os.getenv("OPENAI_ANSWER_MODEL", "gpt-4o-mini")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "missing")) if OpenAI else None

print("Video exists:", VIDEO_PATH.exists())
print("FFmpeg available:", bool(shutil.which("ffmpeg")))
print("OpenAI key set:", bool(os.environ.get("OPENAI_API_KEY")))
print("STT model:", STT_MODEL)
print("Embedding model:", EMBEDDING_MODEL)
print("Optional LLM enabled:", USE_OPENAI_LLM)

# --------------------------------------------------------------------------
# Step 2 - Generate fallback assets if the sample video is missing
# --------------------------------------------------------------------------

step_banner("Step 2 - Generate fallback assets if the sample video is missing", PAUSE)


def create_fallback_frame(path: Path, title: str, subtitle: str, timestamp: int):
    from PIL import Image, ImageDraw, ImageFont

    path.parent.mkdir(parents=True, exist_ok=True)
    W, H = 1280, 720
    img = Image.new("RGB", (W, H), (248, 250, 252))
    draw = ImageDraw.Draw(img)
    try:
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 56)
        font_sub = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 34)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 26)
    except Exception:
        font_title = font_sub = font_small = None
    draw.rectangle([0, 0, W, 92], fill=(30, 41, 59))
    draw.text((50, 28), "Fallback visual record", font=font_small, fill=(255, 255, 255))
    draw.text((W - 170, 28), f"00:{timestamp:02d}", font=font_small, fill=(255, 255, 255))
    draw.rounded_rectangle(
        [90, 150, W - 90, H - 115], radius=22, fill=(255, 255, 255), outline=(210, 215, 225), width=3
    )
    draw.text((130, 210), title, font=font_title, fill=(15, 23, 42))
    draw.text((135, 310), subtitle, font=font_sub, fill=(51, 65, 85))
    draw.text((135, 510), "Used when the MP4 is not available.", font=font_small, fill=(71, 85, 105))
    img.save(path)


if not VIDEO_PATH.exists():
    print("Sample video missing. Creating fallback frame images only.")
    FRAME_DIR.mkdir(parents=True, exist_ok=True)
    fallback_visuals = [
        ("Support Training Video", "Find relevant moments inside video", 0),
        ("Password Reset Issue", "Customer cannot log in; reset email never arrives", 10),
        ("High Priority Ticket", "Agent creates ticket and checks spam filters", 20),
        ("Delivery Delay Case", "Separate case routed to order support", 30),
    ]
    for idx, (title, subtitle, ts) in enumerate(fallback_visuals):
        create_fallback_frame(FRAME_DIR / f"frame_{idx:03d}.jpg", title, subtitle, ts)
else:
    print("Sample video found:", VIDEO_PATH)

# --------------------------------------------------------------------------
# Step 3 - Extract representative frames and audio
# --------------------------------------------------------------------------

step_banner("Step 3 - Extract representative frames and audio", PAUSE)

FRAME_DIR.mkdir(parents=True, exist_ok=True)

if VIDEO_PATH.exists() and shutil.which("ffmpeg"):
    for old in FRAME_DIR.glob("*.jpg"):
        old.unlink()

    subprocess.run(
        ["ffmpeg", "-y", "-i", str(VIDEO_PATH), "-vf", "fps=1/5", str(FRAME_DIR / "frame_%03d.jpg")],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(VIDEO_PATH), "-vn", "-ar", "16000", "-ac", "1", str(AUDIO_PATH)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
else:
    print("Video or FFmpeg unavailable. Continuing with fallback frame and transcript records.")

frame_paths = sorted(FRAME_DIR.glob("*.jpg"))
print("Extracted / available frames:", len(frame_paths))
print("Audio exists:", AUDIO_PATH.exists())

if VIDEO_PATH.exists():
    display(Video(str(VIDEO_PATH), embed=True, width=640))

for path in frame_paths[:4]:
    display(IPImage(filename=str(path), width=320))

# --------------------------------------------------------------------------
# Step 4 - Transcribe audio or use fallback transcript segments
# --------------------------------------------------------------------------

step_banner("Step 4 - Transcribe audio or use fallback transcript segments", PAUSE)

fallback_segments = [
    {
        "start": 0,
        "end": 8,
        "text": "Welcome to the support training video. We will review a customer access issue.",
    },
    {
        "start": 8,
        "end": 20,
        "text": "The customer cannot log into the account because the password reset email never arrives.",
    },
    {
        "start": 20,
        "end": 30,
        "text": "The agent creates a high priority support ticket and asks the customer to check spam filters.",
    },
    {
        "start": 30,
        "end": 38,
        "text": "A delivery delay request is a separate case and should be routed to order support.",
    },
]

transcript_segments = fallback_segments

if AUDIO_PATH.exists() and os.environ.get("OPENAI_API_KEY") and client:
    try:
        with AUDIO_PATH.open("rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model=STT_MODEL,
                file=audio_file,
            )

        text = getattr(transcription, "text", "").strip()
        if text:
            transcript_segments = [{"start": 0, "end": 38, "text": text}]

    except Exception as exc:
        print("Transcription failed. Using fallback transcript segments.")
        print(type(exc).__name__, exc)

segments_df = pd.DataFrame(transcript_segments)
display(segments_df)

# --------------------------------------------------------------------------
# Step 5 - Build timestamped transcript and frame records
# --------------------------------------------------------------------------

step_banner("Step 5 - Build timestamped transcript and frame records", PAUSE)

records = []

for i, seg in enumerate(transcript_segments):
    records.append(
        {
            "id": f"transcript_{i:03d}",
            "modality": "transcript",
            "start": float(seg["start"]),
            "end": float(seg["end"]),
            "content": seg["text"],
            "source": "speech-to-text",
        }
    )

frame_descriptions = [
    "title slide for support training video",
    "support dashboard showing account access and password reset issue",
    "support ticket created with high priority status",
    "delivery delay case shown as a contrast example",
]

for i, frame_path in enumerate(frame_paths[:4]):
    timestamp = i * 5
    records.append(
        {
            "id": f"frame_{i:03d}",
            "modality": "frame",
            "start": float(timestamp),
            "end": float(timestamp),
            "content": frame_descriptions[min(i, len(frame_descriptions) - 1)],
            "source": str(frame_path),
        }
    )

if not frame_paths:
    for i, description in enumerate(frame_descriptions):
        records.append(
            {
                "id": f"frame_{i:03d}",
                "modality": "frame",
                "start": float(i * 10),
                "end": float(i * 10),
                "content": description,
                "source": "fallback_visual_record",
            }
        )

records_df = pd.DataFrame(records)
display(records_df)

# --------------------------------------------------------------------------
# Step 6 - Generate embeddings and create a local index
# --------------------------------------------------------------------------

step_banner("Step 6 - Generate embeddings and create a local index", PAUSE)


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-12, None)


texts = records_df["content"].tolist()
vectorizer = None

if os.environ.get("OPENAI_API_KEY") and client:
    try:
        embedding_response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
        embeddings = np.array([item.embedding for item in embedding_response.data], dtype=np.float32)
        embedding_source = "openai"
    except Exception as exc:
        print("OpenAI embeddings failed. Falling back to local TF-IDF vectors.")
        print(type(exc).__name__, exc)
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer().fit(texts)
        embeddings = vectorizer.transform(texts).toarray().astype(np.float32)
        embedding_source = "tfidf"
else:
    print("No OpenAI API key found. Using local TF-IDF vectors.")
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer().fit(texts)
    embeddings = vectorizer.transform(texts).toarray().astype(np.float32)
    embedding_source = "tfidf"

embeddings = normalize_rows(embeddings)

print("Embedding source:", embedding_source)
print("Embedding matrix shape:", embeddings.shape)
print("First five values of first vector:", embeddings[0][:5])

# --------------------------------------------------------------------------
# Step 7 - Search the video
# --------------------------------------------------------------------------

step_banner("Step 7 - Search the video", PAUSE)


def embed_query(query: str) -> np.ndarray:
    if embedding_source == "openai":
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=[query])
        vector = np.array([response.data[0].embedding], dtype=np.float32)
    else:
        vector = vectorizer.transform([query]).toarray().astype(np.float32)
    return normalize_rows(vector)


query = "Where does the speaker explain the password reset problem?"
query_embedding = embed_query(query)

scores = (query_embedding @ embeddings.T)[0]
records_df["score"] = scores
results = records_df.sort_values("score", ascending=False).head(5)

display(results[["id", "modality", "start", "end", "score", "content", "source"]])

# --------------------------------------------------------------------------
# Step 8 - Fuse nearby transcript and frame evidence
# --------------------------------------------------------------------------

step_banner("Step 8 - Fuse nearby transcript and frame evidence", PAUSE)

best_time = float(results.iloc[0]["start"])
window_seconds = 10

fused_evidence = records_df[
    (records_df["start"] >= best_time - window_seconds) & (records_df["start"] <= best_time + window_seconds)
].sort_values(["start", "modality"])

display(fused_evidence[["id", "modality", "start", "end", "score", "content", "source"]])

# --------------------------------------------------------------------------
# Step 9 - Generate a grounded answer
# --------------------------------------------------------------------------

step_banner("Step 9 - Generate a grounded answer", PAUSE)


def format_time(seconds: float) -> str:
    seconds = int(round(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


context_lines = []
for _, row in fused_evidence.iterrows():
    context_lines.append(
        f"- [{row['modality']}] {format_time(row['start'])}-{format_time(row['end'])}: {row['content']}"
    )

context = "\n".join(context_lines)

prompt = f"""
Answer the user's question using only the evidence below.
Include the most relevant timestamp.
If the evidence is insufficient, say so.

Question: {query}

Evidence:
{context}
""".strip()


def template_answer() -> str:
    password_rows = fused_evidence[
        fused_evidence["content"].str.contains("password|reset|account", case=False, regex=True)
    ]
    if password_rows.empty:
        return "I could not find enough evidence for the password reset question in the retrieved segment."
    first = password_rows.iloc[0]
    return (
        f"The password reset problem is discussed around {format_time(first['start'])}. "
        "The evidence says the customer cannot log into the account because the password reset email never arrives. "
        "The nearby segment also shows that the agent creates a high-priority support ticket and asks the customer to check spam filters."
    )


if USE_OPENAI_LLM and client:
    try:
        response = client.responses.create(model=ANSWER_MODEL, input=prompt)
        answer = response.output_text
    except Exception as exc:
        print("OpenAI answer model failed. Using template answer.")
        print(type(exc).__name__, exc)
        answer = template_answer()
else:
    answer = template_answer()

print(answer)

# --------------------------------------------------------------------------
# Step 10 - Inspect the evidence package
# --------------------------------------------------------------------------

step_banner("Step 10 - Inspect the evidence package", PAUSE)

evidence_package = {
    "query": query,
    "answer": answer,
    "retrieved_evidence": fused_evidence[["id", "modality", "start", "end", "score", "content", "source"]].to_dict(
        "records"
    ),
}

print(json.dumps(evidence_package, indent=2))

# --------------------------------------------------------------------------
# Step 11 - Evaluation checklist
# --------------------------------------------------------------------------

step_banner("Step 11 - Evaluation checklist", PAUSE)

evaluation_questions = pd.DataFrame(
    [
        {"Area": "Retrieval quality", "Question": "Did the top result contain the right topic?"},
        {"Area": "Timestamp accuracy", "Question": "Is the returned time close to the relevant moment?"},
        {"Area": "Grounding", "Question": "Is every answer claim supported by retrieved evidence?"},
        {
            "Area": "Latency",
            "Question": "Which step was slowest: video processing, transcription, embedding, or answering?",
        },
        {"Area": "Storage", "Question": "How many frames and embeddings would a one-hour video create?"},
        {"Area": "Cost", "Question": "Which API calls scale with video length?"},
    ]
)

display(evaluation_questions)
