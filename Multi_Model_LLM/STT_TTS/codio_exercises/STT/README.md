Following learnings can be identified in this STT exercise from Codio:
- class AudioMetadata, adding metadata like sample_rate, channels, duration_ms, size_bytes
- working with the wave class (wave.getframerate(), wave.getnchannels(), getnframes())
- sample_rate = adio.getframerate() to calculate the sample rate from the input wave file
- channels = audio.getnchannels, frame_count = audio.getnframes(), sample_rate=getframerate()
- duration in ms of the wave file based on getnframes()/getframerate()
- validation of audio: supported extension format, supported_sample_rate, unsupported_channels, Invalid duration (min/max) and max file size
- get the file size of a file: size_bytes = path.stat().st_size
- Calculate teh timestamp ranges when segmenten the audio file in smaller pieces, but only note that start and end. Not yet physically splitting up the file (lab.py) , 