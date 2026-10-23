"""Explore cosine similarity and ranking with familiar Python."""

import math


class Record:
    """Store a labelled vector and the embedding space that created it."""

    def __init__(self, source_id: str, label: str, space: str, vector: list[float]) -> None:
        self.source_id = source_id
        self.label = label
        self.space = space
        self.vector = vector


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Return cosine similarity for two compatible, non-zero vectors."""
    if not left or len(left) != len(right):
        raise ValueError("dimension_mismatch")
    left_magnitude = math.sqrt(sum(value * value for value in left))
    right_magnitude = math.sqrt(sum(value * value for value in right))
    if left_magnitude == 0 or right_magnitude == 0:
        raise ValueError("zero_vector")
    dot_product = sum(a * b for a, b in zip(left, right))
    return dot_product / (left_magnitude * right_magnitude)


def rank(query: Record, records: list[Record], k: int = 3) -> list[dict]:
    """Return the highest-scoring records from the query's embedding space."""
    if k <= 0:
        raise ValueError("invalid_k")

    results = []
    for record in records:
        if record.space != query.space:
            continue
        results.append(
            {
                "source_id": record.source_id,
                "label": record.label,
                "score": cosine_similarity(query.vector, record.vector),
            }
        )
    return sorted(
        results,
        key=lambda item: (-item["score"], item["source_id"]),
    )[:k]


def main() -> None:
    """Rank three museum images using small inspectable course vectors."""
    # query = Record("query", "red bicycle", "museum-demo", [1.0, 0.0])\
    query = Record("query", "sleeping cat", "asset-library-demo", [0.0, 1.0])
    images = [
        Record("red-bicycle.png", "red bicycle", "asset-library-demo", [0.95, 0.05]),
        Record("cat-blue-chair.png", "sleeping cat", "asset-library-demo", [0.1, 0.9]),
        Record("beach-sunset.png", "beach sunset", "asset-library-demo", [0.2, 0.8]),
    ]
    for position, result in enumerate(rank(query, images), start=1):
        print(f"{position}. {result['source_id']} ({result['score']:.3f})")


if __name__ == "__main__":
    main()
