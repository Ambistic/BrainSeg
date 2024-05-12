import pickle
import re
from itertools import product
from math import ceil, sqrt

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from geojson import load, dump
from skimage import io
import hashlib


def save_data(data, fn):
    with open(fn, "wb") as f:
        pickle.dump(data, f)


def load_data(fn):
    with open(fn, "rb") as f:
        return pickle.load(f)


def show_batch(batch):
    for i, (a, b) in enumerate(zip(*batch)):
        plt.subplot(2, len(batch[0]), 1 + i)
        plt.imshow(a)
        plt.subplot(2, len(batch[0]), 1 + i + len(batch[0]))
        plt.imshow(b, vmin=0, vmax=1)
    plt.colorbar()


def show_batch_bires(batch):
    # flatten batch
    batch = (*batch[0], batch[1])
    for i, (a, b, c) in enumerate(zip(*batch)):
        plt.subplot(3, len(batch[0]), 1 + i)
        plt.imshow(a)
        plt.subplot(3, len(batch[0]), 1 + i + len(batch[0]))
        plt.imshow(b)
        plt.subplot(3, len(batch[0]), 1 + i + len(batch[0]) * 2)
        plt.imshow(c, vmin=0, vmax=1)
    plt.colorbar()


def show_batch_multires(batch):
    # flatten batch
    batch = (*batch[0], batch[1])
    for i, imgs in enumerate(zip(*batch)):
        nb = len(imgs)
        for j in range(nb - 1):
            plt.subplot(nb, len(batch[0]), 1 + i + len(batch[0]) * j)
            plt.imshow(imgs[j])

        plt.subplot(nb, len(batch[0]), 1 + i + len(batch[0]) * (nb - 1))
        plt.imshow(imgs[-1], vmin=0, vmax=1)
    plt.colorbar()


def rgb_to_multi(arr, table):
    multi_arr = np.zeros(arr.shape[:2] + (len(table),), dtype=bool)
    for i, vec in enumerate(table):
        vec = np.array(vec)
        multi_arr[:, :, i] = (arr == vec).all(axis=2)
    return multi_arr


def multi_to_rgb(arr, table):
    rgb_arr = np.zeros(arr.shape[:2] + (3,), dtype=np.uint8)
    for i, vec in enumerate(table):
        vec = np.array(vec)
        rgb_arr[arr[:, :, i]] = vec
    return rgb_arr


DICT_AREA_COLOR = dict(
    putamen=[20, 200, 100],
    whitematter=[200, 200, 200],
    claustrum=[255, 150, 150],
)


def to_color(areas, dict_area_color=None):
    if dict_area_color is None:
        dict_area_color = DICT_AREA_COLOR

    if isinstance(areas, str):
        areas = [areas]

    return [dict_area_color[area] for area in areas]


def get_slide_size(slide):
    bbox = slide.get_mosaic_bounding_box()
    return bbox.w, bbox.h


def open_image_no_alpha(path):
    img = Image.open(path)
    img = np.asarray(img)
    img = img[:, :, :3]
    return img


def open_mask_binary(path):
    """
    :param path: Path to image (standard format, e.g. png, jpg)
    :return: np.ndarray with ndim = 2 and dtype = bool
    """
    img = Image.open(path)
    img = np.asarray(img)
    img = img[:, :, 0]
    return img > 0


def build_patches(size, downscale, step, template_desc=None):
    """
    Build patches as descriptors
    """
    if template_desc is None:
        template_desc = dict()

    patches = [
        dict(
            **template_desc,
            ori_x=i,
            ori_y=j,
            downscale=downscale,
            step=step,
        )
        for i, j in product(
            range(0, size[0], step * downscale),
            range(0, size[1], step * downscale)
        )
    ]

    return patches


def image_process_plot(img):
    if img.ndim == 2:
        return img
    elif img.ndim == 3:
        if img.shape[2] == 2:
            img = np.concatenate([img, np.zeros(img.shape[:2] + (1,))], axis=2)
        if np.max(img) > 1:
            img = img / 255
        if np.max(img) > 1:
            img = img / np.max(img)

        return img


def show_images(ls, row_size=None):
    size = len(ls)
    if row_size is None:
        row_size = ceil(sqrt(size))

    col_size = ceil(size / row_size)

    for i, img in enumerate(ls):
        plt.subplot(col_size, row_size, 1 + i)
        plt.imshow(image_process_plot(img))


def lmap(*args, **kwargs):
    return list(map(*args, **kwargs))


def getRGBfromI(RGBint):
    blue = RGBint & 255
    green = (RGBint >> 8) & 255
    red = (RGBint >> 16) & 255
    return red, green, blue


def getIfromRGB(rgb):
    red = rgb[0]
    green = rgb[1]
    blue = rgb[2]
    RGBint = (red << 16) + (green << 8) + blue
    return RGBint


def flatten(ls):
    return [x for y in ls for x in y]


def extract_classification_name(x):
    try:
        res = x["name"]
    except Exception:
        res = None

    return res


def get_processing_type(steps, types, number):
    milestones = list(map(int, steps.split()))
    types = types.split()

    if len(milestones) + 1 != len(types):
        raise ValueError("Invalid configuration: Number of milestones and types don't match.\n"
                         "The number of milestone + 1 should be equal the number of types")

    for i in range(len(milestones)):
        if number < milestones[i]:
            return types[i]

    return types[-1]


def read_histo(file_name):
    if file_name.endswith(".geojson"):
        with open(file_name, "r") as f:
            geo = load(f)
        return geo
    raise ValueError(f"Not recognized extension for {file_name}")


def write_histo(geo, file_name):
    if file_name.endswith(".geojson"):
        with open(file_name, "w") as f:
            dump(geo, f)
    else:
        raise ValueError(f"Not recognized extension for {file_name}")


def read_atlas(atlas_file):
    image = io.imread(atlas_file)
    return image[:, :, 0]


def read_txt(file_path):
    with open(file_path) as f:
        file_data = f.readlines()
    return file_data


def write_txt(file_path, data):
    with open(file_path, 'w') as f:
        if isinstance(data, str):
            f.write(data)
        elif isinstance(data, list):
            for line in data:
                f.write(line + '\n')
        else:
            raise TypeError("Data must be either a string or a list of strings.")


def replace_lines_in_file(file_path, pattern, replacement):
    # Read the file
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Replace lines matching the pattern
    new_lines = [re.sub(pattern, replacement, line) for line in lines]

    # Write the modified lines back to the file
    with open(file_path, 'w') as file:
        file.writelines(new_lines)


def hash_file(file_path):
    # Create a hashlib object
    hash_object = hashlib.sha256()

    # Open the file in binary mode and read it in chunks
    with open(file_path, 'rb') as file:
        while True:
            data = file.read(65536)  # Read 64KB at a time (adjust the chunk size as needed)
            if not data:
                break
            hash_object.update(data)

    # Get the hexadecimal representation of the hash
    hash_value = hash_object.hexdigest()

    return hash_value


def filter_index_not_in_list(current_list, discard_list):
    return [x for i, x in enumerate(current_list) if i not in discard_list]


def find_lowest_value_above_threshold(lst, threshold):
    # Initialize a variable to store the lowest value found
    lowest_value = None

    # Iterate through the list
    for value in lst:
        # Check if the current value is higher than the threshold
        if value > threshold:
            # If the lowest_value is None or the current value is lower than the lowest_value,
            # update the lowest_value to the current value
            if lowest_value is None or value < lowest_value:
                lowest_value = value

    # Return the lowest value found
    return lowest_value


def has_larger_range(x, y):
    return min(x) < min(y) and max(x) > max(y)


def get_rolling_slice(ls, start, end):
    if end >= start:
        return ls[start:end]
    else:
        return ls[start:] + ls[:end]
