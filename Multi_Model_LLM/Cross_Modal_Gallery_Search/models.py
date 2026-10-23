"""Small record classes used by the gallery search project."""

class EmbeddingRecord:
    """Store one indexed image and its embedding metadata."""

    def __init__(
        self,
        source: str,
        modality: str,
        encoder: str,
        vector: list[float],
        timestamp_ms: int | None = None,
        label: str = "",
    ) -> None:
        """Initialize the object with the supplied values."""
        self.source = source
        self.modality = modality
        self.encoder = encoder
        self.vector = vector
        self.timestamp_ms = timestamp_ms
        self.label = label


class SearchResult:
    """Represent one ranked image returned to the caller."""

    def __init__(
        self,
        source: str,
        score: float,
        modality: str,
        evidence: str,
        timestamp_ms: int | None = None,
    ) -> None:
        """Initialize the object with the supplied values."""
        self.source = source
        self.score = score
        self.modality = modality
        self.evidence = evidence
        self.timestamp_ms = timestamp_ms

    def to_dict(self) -> dict:
        """Return a JSON-ready representation of this result."""
        return {
            "source": self.source,
            "score": self.score,
            "modality": self.modality,
            "evidence": self.evidence,
            "timestamp_ms": self.timestamp_ms,
        }
