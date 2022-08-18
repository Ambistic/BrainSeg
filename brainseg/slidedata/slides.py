from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import List

from openslide import OpenSlide

OPEN_SLIDE_FORMATS = [
    ".svs",
    ".tif",
    ".ndpi",
    ".vms",
    ".vmu",
    ".scn",
    ".mrxs",
    ".tiff",
    ".svslide",
    ".bif",
    ".czi"
]


@lru_cache
def find_slides_in_dir(slides_root_dir: str) -> List[str]:
    f"""
    Returns a list of files with following extensions : {OPEN_SLIDE_FORMATS}
    """
    slides_root_dir_path = Path(slides_root_dir)

    if slides_root_dir_path.exists():
        slides_files = []

        for root, dirs, files in os.walk(slides_root_dir_path):
            root_path = Path(root)
            for file in files:
                file_path = Path(file)
                if file_path.suffix in OPEN_SLIDE_FORMATS:
                    slides_files.append(str(root_path / file_path))

        return slides_files


@lru_cache
def load_slide(file: [str | Path]) -> OpenSlide:
    return OpenSlide(str(Path(file)))
