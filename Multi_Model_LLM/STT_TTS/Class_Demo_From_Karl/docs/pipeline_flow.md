# STT → RAG → TTS Pipeline

```mermaid
flowchart TD
    A[Input: sample_support_call.m4a] --> B{ffmpeg available?}
    B -->|Yes| C[ffmpeg: resample to 16kHz mono WAV]
    B -->|No| D[librosa/soundfile fallback: resample to 16kHz mono]
    C --> E[sample_16k_mono.wav]
    D --> E
    E --> F[Load audio with librosa<br/>check duration/sample rate]
    F --> G["STT: client.audio.transcriptions.create<br/>(gpt-4o-transcribe)"]
    G --> H[Raw transcript text]

    H --> I["Embed query transcript<br/>(text-embedding-3-small)"]
    J[past_calls: transcripts of<br/>historical support calls] --> K["Embed all past transcripts<br/>(text-embedding-3-small)"]
    I --> L[Normalize embeddings]
    K --> L
    L --> M[cosine_similarity: query vs past]
    M --> N[Rank past calls by score]
    N --> O[Take top 2 most similar<br/>past calls as context]

    H --> P[Build prompt:<br/>new transcript + retrieved context + task instructions]
    O --> P
    P --> Q["LLM: client.responses.create<br/>(gpt-5-mini)"]
    Q --> R[answer: summary + intent + next action
    R --> S[clean_for_tts:<br/>strip markdown/special chars]
    S --> T[spoken_answer]
    T --> U["TTS: client.audio.speech.create<br/>(tts-1, voice=coral)"]
    U --> V[answer_openai.wav]
    V --> W[Play back audio in notebook]

    style A fill:#e1f5fe
    style H fill:#fff3e0
    style R fill:#fff3e0
    style V fill:#e8f5e9
    style W fill:#e8f5e9
```
