from itertools import product
from os.path import basename

import aicspylibczi
import numpy as np
from PIL.Image import Image, open

from brainseg.slide_provider import open_image
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


def create_mask(slide_path, downsample, func, step=512):
    slide = aicspylibczi.CziFile(slide_path)

    bbox = slide.get_mosaic_bounding_box()
    sx, sy = int(bbox.w / downsample), int(bbox.h / downsample)
    mask = np.zeros((sx, sy), dtype=np.bool)

    for i, j in product(range(0, sx, step), range(0, sy, step)):
        img = open_image(slide, (i * downsample, j * downsample), downsample, step)
        pred_mask = func(img)
        # sure ?
        pred_mask = pred_mask.transpose()
        mask[i:i + 512, j:j + 512] = pred_mask[:min(i + 512, sx) - i, :min(j + 512, sy) - j]

    return mask.transpose()


def save_mask_as_image(root, slide_path, image: Image, name="mask"):
    fp = get_slidedata_path(root, slide_path, name + ".png")
    image.save(fp)


def load_mask_as_image(root, slide_path, name="mask"):
    fp = get_slidedata_path(root, slide_path, name + ".png")
    return open(str(fp))


def exists_mask_as_image(root, slide_path, name):
    fp = get_slidedata_path(root, slide_path, name + ".png")
    return fp.exists()
