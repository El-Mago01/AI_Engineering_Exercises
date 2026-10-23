"""Run one configurable lesson scenario."""

import argparse
from lab import embed_gallery, rank


def main() -> None:
    """Run one cross-modal ranking scenario from the command line."""
    parser = argparse.ArgumentParser(description="Explore compatible vector ranking")
    parser.add_argument("--query", default="a red bicycle")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    query, images = embed_gallery(args.query)
    print(f"Query: {args.query}")
    for position, result in enumerate(rank(query, images, args.top_k), start=1):
        print(f"{position}. {result['source_id']} ({result['score']:.3f})")


if __name__ == "__main__":
    main()
