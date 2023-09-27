from math import floor

import numpy as np


def resize_and_pad_center(arr, height=256, width=256, background=0):
    if arr.ndim == 3:
        res = np.zeros((height, width, arr.shape[2]))
    elif arr.ndim == 2:
        res = np.zeros((height, width))
    else:
        raise ValueError("Wrong number of dimension")

    res.fill(background)
    delta_x = floor((height - arr.shape[0]) / 2)
    delta_y = floor((width - arr.shape[1]) / 2)

    res[delta_x: delta_x + arr.shape[0],
        delta_y: delta_y + arr.shape[1]] = arr
    return res
