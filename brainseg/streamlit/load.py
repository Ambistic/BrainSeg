from pathlib import Path

import numpy as np
from PIL import Image

from brainseg.streamlit.manager import check_valid_path


def has_lowres(path, data_name):
    check_valid_path(path)
    fp = Path(path) / data_name / "lowres.png"
    return fp.exists()


def load_mask(path, data_name, mask_name):
    check_valid_path(path)
    fp = Path(path) / data_name / "mask" / mask_name
    image = Image.open(fp)
    return image


def load_lowres(path, data_name):
    check_valid_path(path)
    fp = Path(path) / data_name / "lowres.png"
    image = Image.open(fp)
    return image


def load_image(path, data_name):
    check_valid_path(path)
    fp = Path(path) / data_name / "image.png"
    image = Image.open(fp)
    return image


def load_custom_image(path, data_name, name):
    check_valid_path(path)
    fp = Path(path) / data_name / name
    image = Image.open(fp)
    return image


def load_superpose_mask(path, data_name, mask_name):
    mask = np.asarray(load_mask(path, data_name, mask_name))
    image = np.asarray(load_image(path, data_name))

    new_img = np.maximum(~mask.astype(bool), 0.5) * image
    return new_img.astype(int).astype(np.uint8)


def load_superpose_mask_3(path, data_name, mask_name):
    mask = np.asarray(load_mask(path, data_name, mask_name))
    image = np.asarray(load_image(path, data_name))

    mask = mask.sum(axis=2).astype(bool)
    mask = mask[..., np.newaxis]
    new_img = np.maximum(mask, 0.5) * image
    return new_img.astype(int).astype(np.uint8)


def load_multiply_mask(path, data_name, mask_name):
    mask = np.asarray(load_mask(path, data_name, mask_name))
    image = np.asarray(load_image(path, data_name))

    return ~mask.astype(bool) * image
