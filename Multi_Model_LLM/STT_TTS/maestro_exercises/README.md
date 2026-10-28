The following interesting learnings can be found in the directory: maestro_exercises
- from recorded voice message with a question on an audiofile to a spoken output with answer
- whisper to transcribe the audio file
- embed the transcript #embedding using model="text-embedding-3-small" from openai
- piper with PiperVoice and en_US-lessac-medium.onnx to play the result
- Use openai model "gpt-5-mini" to summarize the transcript, mapped with the top of previous recordings 
- a bad cosine function, which still kind of works... :-)
- ranking a list of tuples with scores and recordings. Sorting is done on the 2nd item in the tuple