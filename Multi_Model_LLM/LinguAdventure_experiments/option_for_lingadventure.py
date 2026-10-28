# TODO: TO MAKE THIS RUN, YOU NEED TO GIVE IT A [pip install "av<19"] first and ofcourse a [pip install faster_whisper]
"""
The following languages are supported:
English	    en
Portuguese	pt
Dutch	    nl
German	    de
Spanish	    es
French	    fr
Italian	    it
Chinese	    zh
Japanese	ja
"""
from faster_whisper import WhisperModel

from pathlib import Path

import sounddevice as sd
import soundfile as sf

# Recording settings
duration = 10            # seconds
sample_rate = 16000      # audio samples per second
channels = 1             # mono

# Save in a "media" folder next to this Python script.
output_path = Path(__file__).resolve().parent / "media" / "my_recording.wav"
output_path.parent.mkdir(parents=True, exist_ok=True)

print("Recording... Speak now!")

# Start recording using the computer's default microphone.
audio = sd.rec(
    int(duration * sample_rate),
    samplerate=sample_rate,
    channels=channels,
    dtype="float32",
)

sd.wait()  # Wait until recording has finished.
print("Recording finished.")

# Write the recorded audio to a WAV file.
sf.write(str(output_path), audio, sample_rate)
print(f"Saved recording to: {output_path}")
# Load a multilingual Whisper model on the CPU.
model = WhisperModel("small", device="cpu", compute_type="int8")

segments, info = model.transcribe(
    "media/my_recording.wav",
    language="nl"  # Portuguese
)

for segment in segments:
    print(segment.text)

print(info)