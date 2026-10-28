"""Retrieve support guidance with real embeddings and generate a grounded answer."""

import math
from dotenv import load_dotenv
import os
from openai import OpenAI
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

MODEL = "gpt-5-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
client = OpenAI(api_key=api_key)

ARTICLES = [
    {
        "source": "account-security.md",
        "text": "For an unfamiliar login, revoke active sessions, reset the password, and contact the security team.",
    },
    {
        "source": "delivery-support.md",
        "text": "Open a carrier investigation when tracking has not updated for more than 48 hours.",
    },
    {
        "source": "export-troubleshooting.md",
        "text": "For a blocked CSV export, record the job ID and retry the export once before escalating.",
    },
]


def cosine(left: list[float], right: list[float]) -> float:
    """Return cosine similarity for two vectors of equal length."""
    dot = sum(a * b for a, b in zip(left, right))
    left_size = math.sqrt(sum(value * value for value in left))
    right_size = math.sqrt(sum(value * value for value in right))
    return dot / (left_size * right_size)


def main() -> None:
    """Retrieve an article and ask the model for a support action."""
    question = "A customer reports an unfamiliar login. What should they do first?"
    # question = "Tracking has not updated for three days. What should support do?" #change the retrieved source compared to other question
    texts = [article["text"] for article in ARTICLES]

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[question, *texts],
    )

    query_vector = response.data[0].embedding
    article_vectors = [item.embedding for item in response.data[1:]]
    scores = [
        cosine(query_vector, vector)
        for vector in article_vectors
    ]

    best_index = max(
        range(len(scores)),
        key=scores.__getitem__,
    )
    article = ARTICLES[best_index]

    answer = client.responses.create(
        model=MODEL,
        instructions=(
            "Answer in one short sentence using only the support article. "
            "End with the source filename in square brackets."
        ),
        input=(
            f"Question: {question}\n"
            f"Article: {article['text']}\n"
            f"Source: {article['source']}"
        ),
    )

    print("Retrieved:", article["source"])
    print("Similarity:", f"{scores[best_index]:.3f}")
    print("Answer:", " ".join(answer.output_text.split())[:180])


if __name__ == "__main__":
    main()