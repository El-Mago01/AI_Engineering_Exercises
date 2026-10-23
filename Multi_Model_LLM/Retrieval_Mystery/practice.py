"""Change a query vector and inspect the museum-image ranking."""

from lab import Record, cosine_similarity, rank


QUERY_VECTOR = [1.0, 0.0]
TOP_K = 3
IMAGE_VECTORS = {
    "red-bicycle.png": [0.95, 0.05],
    "cat-blue-chair.png": [0.1, 0.9],
    "beach-sunset.png": [0.2, 0.8],
    "coffee-books.png": [0.4, 0.6],
}


def main() -> None:
    """Print compact scores followed by the requested top-k ranking."""
    query = Record("query", "museum query", "museum-demo", QUERY_VECTOR)
    records = [
        Record(name, name.removesuffix(".png"), "museum-demo", vector)
        for name, vector in IMAGE_VECTORS.items()
    ]
    for record in records:
        score = cosine_similarity(query.vector, record.vector)
        print(f"{record.source_id}: {score:.3f}")
    print("Top matches:", [item["source_id"] for item in rank(query, records, TOP_K)])


if __name__ == "__main__":
    main()
