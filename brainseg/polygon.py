import warnings

import numpy as np
import shapely
from collections import defaultdict

from shapely import geometry
from shapely.affinity import translate, scale
from shapely.ops import transform
from shapely.validation import make_valid

from brainseg.geo import create_affine_mapping
from brainseg.math import compute_angle
from shapely.geometry import Polygon, MultiPolygon


def translate_to_origin(polygon):
    """
    Translate a Shapely polygon such that its bounding box origin is at (0, 0).

    Parameters:
        polygon (shapely.geometry.Polygon): The Shapely polygon.

    Returns:
        shapely.geometry.Polygon: The translated polygon.
    """
    minx, miny, _, _ = polygon.bounds
    translated_polygon = translate_polygon(polygon, -minx, -miny)

    return translated_polygon


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


def make_polygon_ordering(polygons_outline):
    centroids = np.array([polygon.centroid.coords[0] for polygon in polygons_outline])
    centroids -= centroids.mean(axis=0)
    angles = [compute_angle(x, y) for x, y in centroids]

    return np.argsort(angles)


def validate_no_overflow(polygons, buf=50, raises=True):
    # Check for intersections between polygons
    for i, polygon1 in enumerate(polygons):
        for j, polygon2 in enumerate(polygons[i+1:], start=i+1):
            if polygon1.buffer(buf).intersects(polygon2.buffer(buf)):
                if raises:
                    raise ValueError(f"Polygons {i} and {j} intersect")
                else:
                    warnings.warn(f"Polygons {i} and {j} intersect")


def transform_polygons(ordered_polygons, params):
    transformed_polygons = []

    for poly, param in zip(ordered_polygons, params):
        if param is not None:
            center_x, center_y, flip, rotation_angle, shift_x, shift_y = param
            map_func = create_affine_mapping(flip=flip, rot=rotation_angle,
                                             shift_x=shift_x, shift_y=shift_y,
                                             center_x=center_x, center_y=center_y)
            poly = transform(map_func, poly)

        transformed_polygons.append(poly)
    return transformed_polygons


def match_polygon(source_poly, list_target_poly, contained=None, warn=False):
    if not source_poly.is_valid:
        print("validity issue")
        source_poly = make_valid(source_poly)
    target = None
    area = 0
    for idx, target_poly in enumerate(list_target_poly):
        if not target_poly.is_valid:
            print("validity issue")
            target_poly = make_valid(target_poly)
        new_area = source_poly.intersection(target_poly).area
        if contained is not None:
            is_contained = target_poly.buffer(contained).contains(source_poly)
            if not is_contained:
                continue

        if new_area > area or target_poly.contains(source_poly):
            area = new_area
            target = idx

    if warn and target is None:
        warnings.warn("A polygon did not match any other polygons from the list")

    return target


def get_holes(polygon):
    if isinstance(polygon, Polygon):
        holes = [polygon.interiors]
    elif isinstance(polygon, MultiPolygon):
        holes = [p.interiors for p in polygon.geoms]
    else:
        raise ValueError("Input must be a Shapely Polygon or MultiPolygon.")

    hole_polygons = []
    for interior_rings in holes:
        for ring_coords in interior_rings:
            hole_polygons.append(Polygon(ring_coords))

    return hole_polygons
