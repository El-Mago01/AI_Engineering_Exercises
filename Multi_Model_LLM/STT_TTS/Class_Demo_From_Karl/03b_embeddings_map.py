"""
Embeddings map (slide 8).

Different sentences can mean the same thing even when the words differ. Embed
short support phrases, print cosine similarity between them, show a 2D PCA
scatter labeled with the sentences, and save embeddings_map.png. Retrieval
geometry only — no agent planning or tool calls.
"""

import os
import textwrap
from pathlib import Path

import httpx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

from media_paths import output_dir

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

OUTPUT_DIR = output_dir(__file__)
MAP_PNG = OUTPUT_DIR / "embeddings_map.png"

EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

GROUP_COLORS = {
    "login": "#4e79a7",
    "billing": "#f28e2b",
    "delivery": "#59a14f",
    "microphone": "#e15759",
}


def build_examples() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"label": "login_1", "group": "login", "text": "I cannot access my account"},
            {"label": "login_2", "group": "login", "text": "The password reset email never arrives"},
            {"label": "login_3", "group": "login", "text": "I am locked out and need access today"},
            {"label": "billing_1", "group": "billing", "text": "I was charged twice for my subscription"},
            {"label": "billing_2", "group": "billing", "text": "Please refund the duplicate payment"},
            {"label": "delivery_1", "group": "delivery", "text": "My package is delayed"},
            {"label": "delivery_2", "group": "delivery", "text": "When will my order arrive?"},
            {"label": "mic_1", "group": "microphone", "text": "The microphone is not detected"},
            {"label": "mic_2", "group": "microphone", "text": "My browser cannot use the input device"},
        ]
    )


def main() -> None:
    client = OpenAI(http_client=httpx.Client(trust_env=False))
    print("OpenAI key set:", bool(os.environ.get("OPENAI_API_KEY")))
    print("Embedding model:", EMBEDDING_MODEL)

    examples = build_examples()
    print("\nExample sentences:")
    print(examples.to_string(index=False))

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=examples["text"].tolist(),
    )
    embeddings = np.array([item.embedding for item in response.data], dtype=np.float32)
    print("\nEmbedding shape:", embeddings.shape)

    labels = examples["label"].tolist()
    sim = cosine_similarity(embeddings)
    sim_df = pd.DataFrame(sim, index=labels, columns=labels)
    print("\nCosine similarity:")
    print(sim_df.to_string(float_format="{:.3f}".format))

    coords = PCA(n_components=2, random_state=0).fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(14, 10))
    for group, color in GROUP_COLORS.items():
        mask = examples["group"] == group
        pts = coords[mask.to_numpy()]
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            c=color,
            s=120,
            edgecolors="white",
            linewidths=0.8,
        )

    for i, row in examples.iterrows():
        caption = textwrap.fill(row["text"], width=32)
        ax.annotate(
            caption,
            (coords[i, 0], coords[i, 1]),
            textcoords="offset points",
            xytext=(8, 6),
            fontsize=7,
            ha="left",
            va="bottom",
        )

    ax.set_title("Embedding map (PCA 2D projection)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    fig.savefig(MAP_PNG, dpi=150, bbox_inches="tight")
    print("\nSaved map to:", MAP_PNG)
    plt.show()


if __name__ == "__main__":
    main()
