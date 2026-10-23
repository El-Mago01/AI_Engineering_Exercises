"""Validate images and provide deterministic caption-embedding fixtures."""

import json
from pathlib import Path

SUPPORTED_IMAGES = {".png", ".jpg", ".jpeg", ".webp"}
EMBEDDING_SPACE = "caption-text-embedding-3-small-fixture"


def validate_image(path: str | Path) -> Path:
    """Validate a supported, non-empty image file."""
    image_path = Path(path)
    if image_path.suffix.lower() not in SUPPORTED_IMAGES:
        raise ValueError("unsupported_image_format")
    if not image_path.is_file() or image_path.stat().st_size == 0:
        raise ValueError("missing_or_empty_image")
    return image_path


class FixtureCaptionEmbeddingAdapter:
    """Return repeatable vectors for caption-mediated retrieval tests."""

    def __init__(self, catalog_path: str | Path = "fixtures/catalog.json") -> None:
        self.catalog = json.loads(Path(catalog_path).read_text(encoding="utf-8"))

    def embed_query(self, text: str) -> list[float]:
        """Return the fixture text embedding for a search query."""
        if not text.strip():
            raise ValueError("query_required")
        query = text.lower()
        for keyword, vector in self.catalog["text_queries"].items():
            if keyword in query:
                return vector
        return self.catalog["text_queries"]["default"]

    def embed_image_description(self, path: str | Path) -> list[float]:
        """Return the fixture embedding associated with an image description."""
        image_path = validate_image(path)
        try:
            return self.catalog["images"][image_path.name]["vector"]
        except KeyError as error:
            raise ValueError("image_not_in_fixture_catalog") from error
