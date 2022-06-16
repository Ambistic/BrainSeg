import pickle
import matplotlib.pyplot as plt
import numpy as np


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


def to_color(areas):
    if isinstance(areas, str):
        areas = [areas]

    return [DICT_AREA_COLOR[area] for area in areas]
