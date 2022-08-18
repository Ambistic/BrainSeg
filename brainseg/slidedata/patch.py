from itertools import product
from os.path import basename

import aicspylibczi

from brainseg.slidedata.slidedata import get_slidedata_path
from brainseg.utils import save_data, load_data


def create_patch(slide_path, downsample, step, size):
    slide = aicspylibczi.CziFile(slide_path)

    bbox = slide.get_mosaic_bounding_box()
    sx, sy = int(bbox.w / downsample), int(bbox.h / downsample)

    patches = []
    for i, j in product(range(0, sx, step), range(0, sy, step)):
        patches.append(dict(
            slide_name=basename(slide_path),
            origin=(int(i * downsample), int(j * downsample)),
            downsample=downsample,
            step=step,
            size=size,
        ))
    return patches


def save_patch(root, slide_path, patches, name="patch"):
    fp = get_slidedata_path(root, slide_path, name)
    save_data(patches, fp)


def load_patch(root, slide_path, name="patch"):
    fp = get_slidedata_path(root, slide_path, name)
    return load_data(fp)


def exists_patch(root, slide_path, name="patch"):
    fp = get_slidedata_path(root, slide_path, name)
    return fp.exists()
