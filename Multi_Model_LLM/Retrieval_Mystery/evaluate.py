"""Measure hit-at-k for three labelled museum searches."""

from lab import Record, rank


IMAGES = [
    Record("red-bicycle.png", "red bicycle", "museum-demo", [0.95, 0.05]),
    Record("cat-blue-chair.png", "sleeping cat", "museum-demo", [0.1, 0.9]),
    Record("beach-sunset.png", "beach sunset", "museum-demo", [0.2, 0.8]),
    Record("coffee-books.png", "coffee beside books", "museum-demo", [0.4, 0.6],),
]

EXAMPLES = [
    {"query": [1.0, 0.0], "expected": "red-bicycle.png"},
    {"query": [0.0, 1.0], "expected": "cat-blue-chair.png"},
    {"query": [0.3, 0.7], "expected": "beach-sunset.png"},
    {"query": [0.4, 0.6], "expected": "coffee-books.png"},

]


def top_k_accuracy(k: int) -> float:
    """Return the fraction of expected images found within the top k."""
    hits = 0
    for number, example in enumerate(EXAMPLES, start=1):
        query = Record(f"query-{number}", "labelled query", "museum-demo", example["query"])
        returned = {item["source_id"] for item in rank(query, IMAGES, k)}
        if example["expected"] in returned:
            hits += 1
    return hits / len(EXAMPLES)


if __name__ == "__main__":
    print(f"Top-1 accuracy: {top_k_accuracy(1):.2f}")
