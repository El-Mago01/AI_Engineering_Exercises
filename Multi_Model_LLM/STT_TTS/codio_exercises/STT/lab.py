"""Inspect audio validation, preprocessing and timestamped segmentation."""

from pathlib import Path
import wave


SUPPORTED_EXTENSIONS = {".wav"}


class AudioMetadata:
    """Store the metadata needed before audio reaches a model."""

    def __init__(
        self,
        extension: str,
        sample_rate: int,
        channels: int,
        duration_ms: int,
        size_bytes: int,
    ) -> None:
        """Initialize the object with the supplied values."""
        self.extension = extension
        self.sample_rate = sample_rate
        self.channels = channels
        self.duration_ms = duration_ms
        self.size_bytes = size_bytes


def inspect_wav(path: Path) -> AudioMetadata:
    """Read metadata from a local WAV file without calling a model."""
    with wave.open(str(path), "rb") as audio:
        sample_rate = audio.getframerate()
        channels = audio.getnchannels()
        frame_count = audio.getnframes()
        duration_ms = round((frame_count / sample_rate) * 1000)

    return AudioMetadata(
        extension=path.suffix.lower(),
        sample_rate=sample_rate,
        channels=channels,
        duration_ms=duration_ms,
        size_bytes=path.stat().st_size,
    )


def validate_audio(
    metadata: AudioMetadata,
    max_duration_ms: int = 120_000,
) -> dict:
    """Validate audio metadata before decoding or transcription."""
    if metadata.extension.lower() not in SUPPORTED_EXTENSIONS:
        return {"ok": False, "error": "unsupported_format"}

    if metadata.sample_rate < 8_000 or metadata.sample_rate > 192_000:
        return {"ok": False, "error": "unsupported_sample_rate"}

    if metadata.channels not in (1, 2):
        return {"ok": False, "error": "unsupported_channels"}

    if metadata.duration_ms <= 0 or metadata.duration_ms > max_duration_ms:
        return {"ok": False, "error": "invalid_duration"}

    if metadata.size_bytes <= 0 or metadata.size_bytes > 50_000_000:
        return {"ok": False, "error": "invalid_size"}

    return {"ok": True, "error": None}


def segment_ranges(
    duration_ms: int,
    segment_ms: int,
    overlap_ms: int,
) -> list[dict]:
    """Return timestamp ranges that cover an audio recording."""
    if duration_ms <= 0 or segment_ms <= 0:
        raise ValueError("durations_must_be_positive")

    if overlap_ms < 0 or overlap_ms >= segment_ms:
        raise ValueError("invalid_overlap")

    ranges = []
    step = segment_ms - overlap_ms
    start = 0

    while start < duration_ms:
        end = min(start + segment_ms, duration_ms)
        ranges.append({"start_ms": start, "end_ms": end})

        if end == duration_ms:
            break

        start += step

    return ranges


def transformation_plan(metadata: AudioMetadata) -> list[str]:
    """Return the preprocessing stages required for this recording."""
    actions = ["decode"]

    if metadata.channels == 2:
        actions.append("downmix_to_mono")

    if metadata.sample_rate != 16_000:
        actions.append("resample_to_16000_hz")

    actions.extend(
        [
            "peak_check",
            "segment_with_timestamps",
        ]
    )

    return actions


def main() -> None:
    """Print the metadata, validation result and processing plan."""
    audio_path = Path("media/box-chocs.wav")

    if not audio_path.exists():
        print("Validation:", {"ok": False, "error": "file_not_found"})
        return

    try:
        metadata = inspect_wav(audio_path)
    except (wave.Error, EOFError, OSError):
        print("Validation:", {"ok": False, "error": "invalid_wav_file"})
        return

    print(
        "Audio:",
        {
            "file": str(audio_path),
            "format": metadata.extension.removeprefix(".").upper(),
            "sample_rate_hz": metadata.sample_rate,
            "channels": metadata.channels,
            "duration_seconds": round(metadata.duration_ms / 1000, 2),
            "size_bytes": metadata.size_bytes,
        },
    )
    print("Validation:", validate_audio(metadata))
    print("Processing:", transformation_plan(metadata))
    print(
        "Segments:",
        segment_ranges(metadata.duration_ms, 4_000, 500),
    )


if __name__ == "__main__":
    main()