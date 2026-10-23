"""Build, search and evaluate a caption-mediated image index."""

import math
from pathlib import Path

from image_adapter import (
    EMBEDDING_SPACE,
    SUPPORTED_IMAGES,
    FixtureCaptionEmbeddingAdapter,
    validate_image,
)
from models import EmbeddingRecord, SearchResult


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Calculate cosine similarity for two compatible vectors."""

    inproduct = sum(a * b for a, b in zip(vec_a, vec_b))
    # this creates dot = a1*b1 + a2*b2 + ...., i.e. the internal product of 2 vectors
    vec_a_length = math.sqrt(sum(value * value for value in vec_a))
    # this creates SQRT( (a1)^2 + (a2)^2 + (a3)^2 ...(an)^2 ), i.e. the length of the vectore
    vec_b_length = math.sqrt(sum(value * value for value in vec_b))
    if vec_a_length == 0 or vec_b_length == 0:
        raise ValueError("zero_length")
    if len(vec_a) != len(vec_b):
        raise ValueError("dimension_mismatch")
    return inproduct / (vec_a_length * vec_b_length)


class GalleryIndex:
    """Store description embeddings and provide text or image search."""

    def __init__(self, adapter: FixtureCaptionEmbeddingAdapter) -> None:
        self.adapter = adapter
        self.records: list[EmbeddingRecord] = []

    def add_image(self, path: str | Path) -> EmbeddingRecord:
        """Validate an image and index its supplied description embedding."""
        image_path = validate_image(path)
        vector = self.adapter.embed_image_description(image_path)
        label = self.adapter.catalog["images"][image_path.name]["label"]
        record = EmbeddingRecord(
            source=str(image_path),
            modality="image-description",
            encoder=EMBEDDING_SPACE,
            vector=vector,
            label=label,
        )
        self.records.append(record)
        return record

    def search_vector(self, vector: list[float], *, encoder: str, k: int = 3) -> list[SearchResult]:
        """Return the top compatible images for a query vector.
        Algorithm:
        1. For each image vector,
            0. Check EMBEDDING_SPACE of the image to compare with the received encoder of the query vector
            a. get a score on vector similarity
            b. store this vector similarity and other data in a new  SearchResult object
            c. append the SearchResult object to a new list
        2. sort the list of SearchResults in descending on vector similarity
        3. return a list of the top k SearchResults

        SearchResult definition:
        source: str,
        score: float,
        modality: str,
        evidence: str,
        timestamp_ms: int | None = None,
        """
        if k <= 0:
            raise ValueError("invalid_k")

        if encoder != EMBEDDING_SPACE:
            raise ValueError("incompatible_embedding_space")

        search_results = []

        for image_record in self.records:
            if (image_record.encoder != encoder or
                    len(vector) != len(image_record.vector) or
                    len(vector) == 0):
                continue
            result=SearchResult(
                source=image_record.source,
                score=round(cosine_similarity(vector, image_record.vector), 6),
                modality=image_record.modality,
                evidence="",
                timestamp_ms=image_record.timestamp_ms
            )
            search_results.append(result)
        search_results.sort(key=lambda result: (-result.score, result.source))
        return search_results[:k]





    def search_text(self, query: str, k: int = 3) -> list[SearchResult]:
        """Embed a text query and return the nearest image descriptions."""
        vector = self.adapter.embed_query(query)
        return self.search_vector(vector, encoder=EMBEDDING_SPACE, k=k)

    def search_image(self, path: str | Path, k: int = 3) -> list[SearchResult]:
        """Use an image's description embedding as a similarity query."""
        vector = self.adapter.embed_image_description(path)
        return self.search_vector(vector, encoder=EMBEDDING_SPACE, k=k)


def evaluate_top_k(index: GalleryIndex, examples: list[dict], k: int = 1) -> float:
    """Measure how often the expected image appears within the top ``k``.
    Algorithm:
    1. From SearchResults

    """
    if not examples:
        raise ValueError("examples_required")
    hits = 0
    for example in examples:
        results = index.search_text(example["query"], k)
        filenames = {
            Path(result.source).name
            for result in results
        }
        if example["expected"] in filenames:
            hits += 1
    return hits / len(examples)

def build_default_index() -> GalleryIndex:
    """Build an index from the bundled image directory."""
    index = GalleryIndex(FixtureCaptionEmbeddingAdapter())
    for path in sorted(Path("media/images").glob("*")):
        if path.suffix.lower() in SUPPORTED_IMAGES:
            index.add_image(path)
    return index

def main():
    print(cosine_similarity([1.0, 0.0], [1.0, 0.0]))
    print(cosine_similarity([1.0, 0.0], [0.0, 1.0]))
    aGalleryIndex = build_default_index()
    result = aGalleryIndex.search_vector([0.95,0.05,0.0,0.0], encoder=EMBEDDING_SPACE, k=3)
    print(result)

if __name__ == "__main__":
    main()