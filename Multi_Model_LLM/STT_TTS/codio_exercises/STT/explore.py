"""Run one configurable lesson scenario."""

import argparse
import json

from lab import AudioMetadata, segment_ranges, transformation_plan, validate_audio


def main() -> None:
    """Run one audio preprocessing scenario from the command line."""
    parser = argparse.ArgumentParser(description="Explore audio segmentation")
    parser.add_argument("--duration", type=int, default=9200)
    parser.add_argument("--segment", type=int, default=4000)
    parser.add_argument("--overlap", type=int, default=500)
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--channels", type=int, default=2)
    args = parser.parse_args()
    metadata = AudioMetadata(".wav", args.sample_rate, args.channels, args.duration, 400000)
    result = {
        "validation": validate_audio(metadata),
        "transformation_plan": transformation_plan(metadata),
    }
    try:
        result["ranges"] = segment_ranges(args.duration, args.segment, args.overlap)
    except ValueError as error:
        result["segmentation_error"] = str(error)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
