"""Change audio metadata and inspect the required processing stages."""

from lab import AudioMetadata, segment_ranges, transformation_plan, validate_audio


SAMPLE_RATE = 48_000
CHANNELS = 1
DURATION_MS = 9_200


def main() -> None:
    """Print validation, transformations and timestamped segments."""
    metadata = AudioMetadata(".wav", SAMPLE_RATE, CHANNELS, DURATION_MS, 400_000)
    print("Validation:", validate_audio(metadata))
    print("Processing:", transformation_plan(metadata))
    print("Segments:", segment_ranges(DURATION_MS, 4_000, 500))


if __name__ == "__main__":
    main()
