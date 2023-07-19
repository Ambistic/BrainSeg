import numpy as np


def create_3d_histogram(data, shape, bin_starts, bin_stops, weights=None):
    """
    Creates a 3D histogram from the input data and returns an array with the specified shape
    and bin coordinates.

    Parameters:
    data (ndarray): The input data array.
    shape (tuple): The desired shape of the output histogram array.
    bin_starts (tuple): The start coordinates of the bins for each dimension.
    bin_stops (tuple): The stop coordinates of the bins for each dimension.

    Returns:
    ndarray: A 3D histogram array with the specified shape and bin coordinates.
    """
    bins = tuple(
        np.linspace(start, stop, num=shape[i] + 1) for i, (start, stop) in enumerate(zip(bin_starts, bin_stops))
    )
    hist, _ = np.histogramdd(data, bins=bins, weights=weights)
    return hist


def bisect_left(a, x, lo=0, hi=None):
    """Return the index where to insert item x in list a, assuming a is sorted.

    The return value i is such that all e in a[:i] have e < x, and all e in
    a[i:] have e >= x.  So if x already appears in the list, i points just
    before the leftmost x already there.

    Args:
        a (list): A sorted list of comparable elements.
        x: The value to be inserted into the list.
        lo (int, optional): The lower bound of the search interval. Defaults to 0.
        hi (int, optional): The upper bound of the search interval. Defaults to None.

    Returns:
        int: The index where x should be inserted in the list a.
    """
    if hi is None:
        hi = len(a)
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return lo


def interpolate(indices, values):
    # Define the interpolation function
    def interp(x):
        i = bisect_left(indices, x)
        if i == 0:
            return values[0]
        elif i == len(indices):
            return values[-1]
        else:
            x0, x1 = indices[i-1], indices[i]
            y0, y1 = values[i-1], values[i]
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)

    return interp
