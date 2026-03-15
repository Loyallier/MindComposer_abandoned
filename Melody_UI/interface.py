import os
import time
import uuid
import shutil
import logging
import subprocess
import inspect
from typing import Any, Dict, List, Optional, Union

from pipeline import (
    BASE_DIR,
    OUTPUT_DIR,
    LOG_DIR,
    VALID_STYLES,
    predict_harmony_from_midi,
    detect_style_from_midi,
    render_music,
    midi_to_mp3,
    midi_to_musicxml,
    generate_melody_midi,
    generate_song,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "interface.log"), encoding="utf-8"),
        logging.StreamHandler(),
    ],
)

__all__ = [
    "BASE_DIR",
    "OUTPUT_DIR",
    "LOG_DIR",
    "VALID_STYLES",
    "predict_harmony_from_midi",
    "detect_style_from_midi",
    "render_music",
    "midi_to_mp3",
    "midi_to_musicxml",
    "generate_melody_midi",
    "generate_song",
]
