"""Command-line entry point for the cross-modal gallery project.
Project Brief
A communications team has four images that staff need to search using ordinary sentences.
For example, the query a sleeping pet should return the cat image first. The
system uses gpt-5-mini to create a factual description of each image, then
embeds descriptions and queries with text-embedding-3-small.
You will complete three functions in gallery_index.py:
    cosine_similarity compares two vectors.
    search_vector ranks the indexed images.
    evaluate_top_k measures whether the correct image was retrieved.
Everything else is supplied so you can focus on multimodal retrieval.

gallery_index.py       edit this file
image_adapter.py       supplied image validation and fixture encoder
models.py              supplied result classes
gallery_search.py      supplied command-line interface
media/images/          the four visible gallery images
fixtures/              repeatable vectors and labelled evaluation queries
"""

import argparse
import json
from pathlib import Path

from gallery_index import (
    GalleryIndex,
    build_default_index,
    cosine_similarity,
    evaluate_top_k,
)
from image_adapter import (
    EMBEDDING_SPACE,
    SUPPORTED_IMAGES,
    FixtureCaptionEmbeddingAdapter,
    validate_image,
)
from models import EmbeddingRecord, SearchResult


def main() -> None:
    """Run gallery search from the command line."""
    parser = argparse.ArgumentParser(description="Cross-modal gallery search")
    parser.add_argument("--query", default="a bicycle")
    parser.add_argument("--image")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    index = build_default_index()
    if args.check:
        examples = json.loads(
            Path("fixtures/evaluation.json").read_text(encoding="utf-8")
        )
        accuracy = evaluate_top_k(index, examples, 1)
        if accuracy != 1.0:
            print("CHECK FAIL: complete ranking and evaluation first")
            raise SystemExit(1)
        print("CHECK PASS: labelled queries retrieve the expected image")
        return

    results = (
        index.search_image(args.image, args.top_k)
        if args.image
        else index.search_text(args.query, args.top_k)
    )
    print(json.dumps([result.to_dict() for result in results], indent=2))


if __name__ == "__main__":
    main()
