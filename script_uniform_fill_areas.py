#!/usr/bin/env python
# coding: utf-8
import argparse
import traceback

from shapely.geometry import Point, shape
from pathlib import Path
from geojson import load as geo_load, dump as geo_dump
import numpy as np
from brainseg import geo

from brainseg.config import fill_with_config
from brainseg.path import build_path_from_mask


def fill_polygon_with_points(polygon, spacing):
    """
    Fill a Shapely polygon with points uniformly using a grid distribution.

    Parameters:
    polygon (Polygon): A Shapely polygon to be filled with points.
    spacing (float): The distance between adjacent points in the grid.

    Returns:
    list: A list of Shapely Points within the polygon.
    """
    # Get the bounds of the polygon
    minx, miny, maxx, maxy = polygon.bounds

    # Generate points within the bounding box with the specified spacing
    x_coords = np.arange(minx, maxx, spacing)
    y_coords = np.arange(miny, maxy, spacing)

    points_within_polygon = []

    # Iterate over the grid points
    for x in x_coords:
        for y in y_coords:
            point = Point(x, y)
            # Check if the point is within the polygon
            if polygon.contains(point):
                points_within_polygon.append(point)

    return points_within_polygon


def run(area_file, output_file):
    with open(area_file, "r") as f:
        geo_feats = geo_load(f)

    all_points = []
    for feat in geo_feats["features"]:
        if feat is None:
            continue

        if feat.get("geometry", dict()).get("type") != "Polygon":
            continue
        # exclude list
        if feat.get("properties", dict()).get("name") in ["pial", "WM", "none"]:
            continue

        name = feat.get("properties").get("name")
        print(name)

        polygon = shape(feat["geometry"])
        points = fill_polygon_with_points(polygon, spacing=100)  # spacing is arbitrary
        print(len(points))

        for point in points:  # maybe create a multipoint instead ?
            geo_feats["features"].append(geo.create_qupath_single_point(point.x, point.y, point_name=name))

    with open(output_file, "w") as f:
        geo_dump(geo_feats, f)


def main(args):
    for slice_id in range(args.start, args.end, args.step):
        area_file = build_path_from_mask(args.atlas_areas_dir, slice_id, args.labeled_area_mask, zfill=True)
        output_file = build_path_from_mask(args.annotations_dir, slice_id, args.merged_annotations_mask, zfill=True)

        if not Path(area_file).exists():
            print("Skip", slice_id)
            continue

        try:
            run(area_file, output_file)
        except Exception:
            print(traceback.format_exc())
        else:
            print("Slice", slice_id, "run properly")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=Path, default=Path("/media/tower/LaCie/REGISTRATION_PROJECT/"
                                                                  "TMP_PIPELINE_M148/config.ini"))
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--labeled_area_mask", type=str, default=None)
    parser.add_argument("--atlas_areas_dir", type=Path, default=None)
    parser.add_argument("--merged_annotations_mask", type=str, default=None)
    parser.add_argument("--annotations_dir", type=Path, default=None)

    args_ = fill_with_config(parser)

    main(args_)
