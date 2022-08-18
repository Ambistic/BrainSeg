from os.path import basename
from pathlib import Path


def get_slidedata_path(root, slide_path, datatype):
    (Path(root) / basename(slide_path)).mkdir(exist_ok=True, parents=True)
    return Path(root) / basename(slide_path) / datatype
