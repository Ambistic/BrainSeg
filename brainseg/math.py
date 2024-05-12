import math

import numpy as np


def compute_angle(x, y):
    angle = math.atan2(y, x)
    if angle < 0:
        angle += 2 * math.pi
    return math.degrees(angle)


def distance(a, b):
    return np.sqrt(np.sum((np.array(a) - np.array(b))**2, axis=-1))
