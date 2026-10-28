"""
From Maestro Lesson on Building RAG:
Building RAG -> Building RAG -> Building a Simple RAG Pipeline


Let’s keep this very concrete, Martin 👇

In this lesson, we’ll wire up a minimal working RAG pipeline in Python. At a high level, we need just four things:

1. Documents (text chunks you already know how to create)
2. An embedding model (turns text into vectors)
3. A vector store (stores those vectors + lets us do similarity search)
4. An LLM (takes the retrieved text + question and generates an answer)

End-to-end flow we’ll practice:
ingest → embed + store → query embed → retrieve → build prompt → generate.

Layer 0: PDF → text
Layer 1: RAG pipeline (embed → store → retrieve → generate)
"""
from pypdf import PdfReader
import math
import requests

OLLAMA_HOST = "http://192.168.1.75:11434"  # change this
# EMBED_MODEL = "hf.co/bartowski/gemma-1.1-2b-it-GGUF:latest"  # or qwen3:8b etc.
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "hf.co/bartowski/gemma-1.1-2b-it-GGUF:latest"  # you can use the same or pick qwen3:8b




def embed(text: str):
    url = f"{OLLAMA_HOST}/api/embed"
    payload = {
        "model": EMBED_MODEL,
        "input": text,
    }
    resp = requests.post(url, json=payload)
    resp.raise_for_status()
    data = resp.json()
    return data["embeddings"][0]  # this should be a list[float]


def load_pdf_text(path):
    reader = PdfReader(path)
    pages_text = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages_text.append(text.strip())
    full_text = "\n\n".join(pages_text)
    return full_text

def chunk_text(text, chunk_size=500):
    """
    Chunking based on fixed size
    :param text:
    :param chunk_size:
    :return:
    """
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i+chunk_size].strip()
        if chunk:
            chunks.append(chunk)
    return chunks

def build_vector_store(chunks):
    store = []
    for chunk in chunks:
        store.append({
            "chunk": chunk,
            "embedding": embed(chunk)
        })
    return store

# def embed(text, dim=64):
    """
    A very simple and silly, deterministic "embedding". IT does not capture meaning. In real embedding models,
    sentences with similar meaning land near each other in vector space, even if they use different words.

    Here’s what that code does,
    this loop:
    turns the string into a fixed-length vector of size dim
    where each entry is a sum of character codes that landed in that “bucket.”

    step by step:
    enumerate(text) gives you:
    i = position of the character (0, 1, 2, 3, …)
    ch = the actual character at that position ('a', 'b', ' ', etc.)

    ord(ch) converts a character to its Unicode code point (an integer).
    e.g. ord('A') == 65, ord(' ') == 32.

    i % dim wraps the position index into the range 0 … dim-1.
    So if dim = 64, all characters at positions 0, 64, 128, … will affect vec[0];
    positions 1, 65, 129, … affect vec[1], and so on.

    vec[i % dim] += ord(ch) means:
    “Add the numeric value of this character into one bucket of a fixed-size vector,
    based on its position modulo dim.”

    Then we normalized it to unit length so magnitudes don’t explode.

    :param text:
    :param dim:
    :return:
    """
    # vec = [0.0] * dim
    # for i, ch in enumerate(text):
    #     vec[i % dim] += ord(ch)
    # # L2 normalize
    # norm = math.sqrt(sum(x*x for x in vec)) or 1.0
    # return [x / norm for x in vec]

def cosine(a, b):
    return sum(x*y for x, y in zip(a, b))

def generate_answer(question: str, contexts: list[str]) -> str:
    url = f"{OLLAMA_HOST.rstrip('/')}/api/chat"

    system_prompt = (
        "You are a helpful assistant that answers questions using ONLY the provided context. "
        "If the answer is not in the context, say you don't know."
    )

    context_text = "\n\n---\n\n".join(
        context.strip() for context in contexts if context and context.strip()
    )

    if not context_text:
        return "I don't know—the provided context is empty."

    print("The prompt send to the chat LLM:", context_text)
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"Context:\n{context_text}\n\n"
                f"Question: {question}\n\n"
                "Answer clearly based only on the context."
            ),
        },
    ]

    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "stream": False,
    }

    resp = requests.post(url, json=payload, timeout=300)
    print("URL:", resp.url)
    print("Status:", resp.status_code)
    print("Body:", resp.text[:1000])
    resp.raise_for_status()
    data = resp.json()

    return data["message"]["content"]

def retrieve(query, store, k=5):
    q_vec = embed(query)
    scored = []
    for idx, item in enumerate(store):
        score = cosine(q_vec, item["embedding"])
        scored.append((score, item["chunk"], idx))
    reranked = rerank(query, scored)
    return reranked[:k]

def rerank(question, scored_chunks):
    """
    Bias toward intro/summary sections (cheap heuristic)
    For a question like “Waar gaat deze training over?”, the answer is almost always near:

    the start of the document
    sections with words like “doel”, “inhoud”, “programma”, “opleiding”, “beschrijving”
    You can gently nudge the ranking with a tiny heuristic (quick fix):
    :param question:
    :param scored_chunks:
    :return:
    """
    # scored_chunks: list of (score, chunk, index)
    boosted = []
    for score, chunk, idx in scored_chunks:
        bonus = 0.0

        # prefer earlier chunks
        if idx < 10:
            bonus += 0.05

        # prefer chunks that look like intro/description
        keywords = ["doel", "inhoud", "opleiding", "training", "programma", "beschrijving"]
        if any(kw.lower() in chunk.lower() for kw in keywords):
            bonus += 0.05

        boosted.append((score + bonus, chunk, idx))

    boosted.sort(reverse=True, key=lambda x: x[0])
    return boosted

"""
we need to turn  chunks into vectors and store them in a tiny in-memory vector store.
At a high level, we’ll do this:


1. Define an embedding function: text -> list[float]
2. Build an in-memory store: a Python list of dicts like:
[
    {"chunk": "...", "embedding": [0.12, ...]},
    ...
]
]
"""

if __name__ == "__main__":
    pdf_path = "../docs/opleidingsgids_zijnscoaching.pdf"  # change this
    text = load_pdf_text(pdf_path)
    print(len(text))  # preview first 2000 chars
    chunks = chunk_text(text, chunk_size=800)
    print(f"Got {len(chunks)} chunks")

    store = build_vector_store(chunks)
    question = "Waar gaat deze training over?"
    top = retrieve(question, store, k=5)
    for score, chunk, _ in top:
        print("SCORE:", score)
        print(chunk[:600])
        print("\n" + "-" * 80 + "\n")
    top_chunks = [chunk for score, chunk, _ in top]

    answer = generate_answer(question, top_chunks)
    print("ANSWER:\n", answer)

