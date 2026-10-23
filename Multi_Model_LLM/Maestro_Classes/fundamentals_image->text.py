"""
https://app.maestro.org/learn/6a7217ef3eccbe038f09e6b0-1/6a7217c57f8b056f7efc6658-1/6a7217dacc911d99cd4e437e

we’ll work with:


1 image → turned into 1 image embedding
Several captions → each turned into its own text embedding
Then we’ll compare them with cosine similarity and see which caption scores highest.

Python script :
loads "man with red balloon.jpeg"
defines 3–4 captions
computes cosine similarity between the image embedding and each caption embedding
prints: caption | similarity score
"""


"""
get_image_embedding("man with red balloon.jpeg")
→ gives you something like a Python list or array:
[-0.02, 0.13, ..., 0.41] (hundreds or thousands of numbers)

get_text_embedding("a man with a red balloon")
→ gives another vector, same length, in the same space.
Because both are in the same embedding space, we can use cosine similarity to ask:
“How much does the direction of the image vector align with the caption vector?”
Higher cosine similarity → better match (within this set of captions).
We’ll treat get_image_embedding and get_text_embedding as black-box helpers that just return those vectors.
"""

import math

# --- 1. Fake embedding functions (stand-ins for real models) ---

def get_image_embedding(image_path):
    """
    Pretend to turn the image into a vector.
    For now, we just return a fixed vector that we'll treat as
    "man with red balloon".
    """
    return [0.9, 0.1, 0.2]


def get_text_embedding(text):
    """
    Pretend to turn text into a vector.
    We hand‑craft these so we can see meaningful similarity differences.
    """
    text = text.lower()

    if "man" in text and "red" in text and "balloon" in text:
        # Very good match
        return [0.92, 0.08, 0.15]
    elif "man" in text and "balloon" in text:
        # Good, but missing the color "red"
        return [0.85, 0.15, 0.25]
    elif "child" in text and "balloon" in text:
        # Related (balloon) but wrong subject
        return [0.5, 0.4, 0.1]
    elif "city" in text or "street" in text:
        # Different scene
        return [0.1, 0.9, 0.3]
    else:
        # Generic mismatch
        return [0.0, 0.0, 1.0]


# --- 2. Cosine similarity helper ---

def cosine_similarity(v1, v2):
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


# --- 3. Image path and candidate captions ---

image_path = "man with red balloon.jpeg"

captions = [
    "a man with a red balloon",
    "a man with a blue balloon",
    "a man holding a balloon",
    "a child holding a blue balloon",
    "a man walking in a busy city street",
]

# --- 4. Compute embeddings and similarities ---

image_emb = get_image_embedding(image_path)

results = []

for caption in captions:
    text_emb = get_text_embedding(caption)
    sim = cosine_similarity(image_emb, text_emb)
    results.append((caption, sim))

# --- 5. Sort captions by similarity (highest first) and print ---

results.sort(key=lambda x: x[1], reverse=True)

print("Image:", image_path)
print("\nCaptions ranked by cosine similarity:\n")
for caption, score in results:
    print(f"{score:.3f}  |  {caption}")