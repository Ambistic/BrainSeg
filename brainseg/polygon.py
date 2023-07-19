import numpy as np
import shapely
from collections import defaultdict

from shapely import geometry
from shapely.affinity import translate, scale


def translate_polygon(polygon, x, y):
    translated_polygon = translate(polygon, xoff=x, yoff=y)
    return translated_polygon


def rescale_polygon(polygon, scale_factor):
    rescaled_polygon = scale(polygon, xfact=scale_factor, yfact=scale_factor, origin=(0, 0))
    return rescaled_polygon


def is_outer(polygon):
    """
    Test if a polygon is outer.

    Parameters:
        polygon (list): The polygon as a list of vertices, in clockwise or counter-clockwise order.

    Returns:
        bool: True if the polygon is outer, False if it is a hole.
    """
    # Compute the signed area of the polygon
    area = 0.0
    n = len(polygon)
    for i in range(n):
        j = (i + 1) % n
        area += polygon[i][0] * polygon[j][1] - polygon[j][0] * polygon[i][1]

    # The polygon is outer if the signed area is positive
    return area > 0


def find_parent_id(poly_inner, polygon_outer):
    idx = -1
    area = np.inf
    p_i = shapely.geometry.Polygon(poly_inner)
    for i, poly_outer in enumerate(polygon_outer):
        p_o = shapely.geometry.Polygon(poly_outer)
        if not p_i.within(p_o):
            continue

        if p_o.area < area:
            area = p_o.area
            idx = i

    return idx


def build_polygon_correspondences(polygon_list):
    polygon_outer = list(filter(is_outer, polygon_list))
    polygon_inner = list(filter(lambda x: not is_outer(x), polygon_list))

    correspondences = defaultdict(list)

    for poly_inner in polygon_inner:
        idx = find_parent_id(poly_inner, polygon_outer)
        correspondences[idx].append(poly_inner)

    return polygon_outer, correspondences


def generate_patch_polygon(origin, size, downsample):
    # Calculate the scaled size based on the downsample level
    scaled_size = size * downsample

    # Calculate the coordinates of the square vertices
    x1, y1 = origin
    x2, y2 = x1 + scaled_size, y1
    x3, y3 = x1 + scaled_size, y1 + scaled_size
    x4, y4 = x1, y1 + scaled_size

    # Create the square polygon
    square = geometry.Polygon([(x1, y1), (x2, y2), (x3, y3), (x4, y4)])

    return square
