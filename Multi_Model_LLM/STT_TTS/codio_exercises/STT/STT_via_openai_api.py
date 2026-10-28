"""
The recording is speech stored as audio samples. A speech-to-text pipeline must turn those samples into words while preserving where the words occurred.

WAV file -> validate -> decode -> transcribe -> timestamped text
Before sending the file to a model, Python can check its sample rate, channels and duration:
"""

import wave


# This catches damaged or unexpected input before an API call costs time or money.
with wave.open("media/box-chocs.wav", "rb") as audio:
    sample_rate = audio.getframerate()
    channels = audio.getnchannels()
    duration = audio.getnframes() / sample_rate