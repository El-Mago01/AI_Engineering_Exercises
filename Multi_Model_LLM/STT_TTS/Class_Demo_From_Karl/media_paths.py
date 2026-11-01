"""
Shared media locations for the demo scripts.

  media/inputs/              - sample files the demos read
  media/outputs/<script>/    - files a demo generates (safe to delete)

Paths are anchored to this file, so the demos work from any directory.
"""

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
MEDIA_DIR = PROJECT_DIR / "media"
INPUTS_DIR = MEDIA_DIR / "inputs"
OUTPUTS_DIR = MEDIA_DIR / "outputs"

SUPPORT_CALL_AUDIO = INPUTS_DIR / "sample_support_call.m4a"
SUPPORT_VIDEO = INPUTS_DIR / "sample_multimodal_support_video.mp4"


def output_dir(name: str) -> Path:
    """Return the output folder for a demo, creating it if needed.

    Pass __file__ from a script, or a plain name such as
    "04_building_multimodal_apps" from a notebook.
    """
    path = OUTPUTS_DIR / Path(name).stem
    path.mkdir(parents=True, exist_ok=True)
    return path
