"""
Maestro exercise for Speech, Audio and Language Chapter, Session: Reasoning over audio.
“Given one transcript, find which other transcripts (recordings) are most similar.”
You already have a transcript variable (right now it’s a dummy string from transcribe_audio).
You’ll create a small list of other “recordings”, each represented as text.
You’ll: turn the transcript into a vector (with a fake embed_text)
turn each other recording into a vector
compute a fake similarity score
sort and print the most similar ones

"""
import os
from openai import OpenAI
from typing import List
from dotenv import load_dotenv
import whisper
import wave
import subprocess
from piper import PiperVoice

model=whisper.load_model("base")

load_dotenv()

audio_file = "audio/Team meeting q4.wav"  # or .mp3

api_key = os.getenv("OPENAI_API_KEY")
MODEL_NAME="gpt-5-mini"
client = OpenAI(api_key=api_key)

def embed_text(text: str) -> List[float]:
    # TODO: replace with real embedding model later
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return [item.embedding for item in response.data]

SYSTEM_PROMPT="""
You are and administrative support employee and you summarize the user's transcript in a few words 
Please output the text in such a way that it can be read by a text-to-speech bot. Make the output at least 2 lines.
See the transcript here:.
"""

def similarity(a: List[float], b: List[float]) -> float:
    # Super fake similarity: negative distance
    return -sum((x - y) ** 2 for x, y in zip(a, b))

recordings = [
    "Team meeting about Q4 revenue and product roadmap.",
    "Casual chat about weekend plans and hobbies.",
    "Discussion of product launch timeline and marketing strategy.",
    "Technical review of system architecture and performance."
]
recording_records: List [List[tuple]]=[]

def get_system_prompt(transcript: str) -> str:
    return (SYSTEM_PROMPT + "\n\n*********User Transcript:\n\n" + transcript +
            "\n\n*******Past recordings:*******]\n\n Please look into the user content")

def transcribe_audio(path: str) -> str:
    result = model.transcribe(audio_file, language="en")
    print(result["text"])
    return result["text"]

def just_say_it(text:str) -> bool:
    voice = PiperVoice.load("en_US-lessac-medium.onnx")
    with wave.open("reply.wav", "wb") as audio_file:
        voice.synthesize_wav(text, audio_file)
    subprocess.run(["aplay", "reply.wav"], check=True)
    return True

def summarize(records: List[str], transcript:str) -> str:
    user_prompt = "\n".join([record[0] for record in records])
    running_input = [
        {"role": "system", "content": get_system_prompt(transcript)},
        {"role": "user", "content": user_prompt},
    ]
    print("PROMPT:", running_input)
    response = client.responses.create(
        model=MODEL_NAME,
        input=running_input,
    )
    return response.output_text
print("START")
transcript = transcribe_audio(audio_file)
print("STOP")
print("START")
embedded_transcript = embed_text(transcript)[0]
embedded_recordings = embed_text(recordings)
for index in range(len(recordings)):
    recording_text = recordings[index]
    similarity_score = similarity(embedded_transcript, embedded_recordings[index])
    recording_record = (recording_text, similarity_score)
    recording_records.append(recording_record)

recording_records.sort(key=lambda x: x[1], reverse=True)
summary=summarize(recording_records[:2],transcript)
print("STOP")
print("START")
just_say_it(summary)
print("STOP")
print(f"Summary:\n{summary}")

