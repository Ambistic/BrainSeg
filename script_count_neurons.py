import argparse
import os.path
from functools import reduce
from itertools import product
from pathlib import Path
import geopandas as gpd
import pandas as pd
from matplotlib import pyplot as plt
from shapely.geometry import Polygon, MultiPolygon, MultiPoint, Point
from collections import defaultdict

from brainseg.config import fill_with_config
from brainseg.misc.volume_ops import interpolate_dicts
from brainseg.path import build_path_histo
from brainseg.utils import read_histo, read_txt


def get_classification_name(row):
    try:
        cl = row["classification"]["name"]
    except:
        cl = None
    return cl


def get_point_name(row):
    try:
        cl = row["classification"]["name"]
    except:
        cl = None
    return cl


def parse_one(args, i):
    file = build_path_histo(args.mri_atlas_dir, i, args.merged_annotations_mask)
    if not os.path.exists(file):
        return None

    df = gpd.read_file(file)

    polygons = []
    points = []

    for i, x in df.iterrows():
        if isinstance(x.geometry, (Point, MultiPoint)):
            if get_point_name(x) is not None:
                points.append(x)
        elif isinstance(x.geometry, (Polygon, MultiPolygon)):
            if get_classification_name(x) is not None:
                polygons.append(x)

    print(len(polygons), len(points))
    matrix = defaultdict(lambda: defaultdict(int))

    for poly, point in product(polygons, points):
        if isinstance(point, MultiPoint):
            count = sum(1 for p in point.geometry.geoms if p.within(poly.geometry))
        else:
            count = 1 if point.geometry.within(poly.geometry) else 0
        # in_polygon = poly.geometry.contains(point.geometry)
        matrix[get_point_name(point)][get_classification_name(poly)] += count

    for point in points:
        if isinstance(point, MultiPoint):
            count = sum(1 for _ in point.geometry.geoms)
        else:
            count = 1
        matrix[get_point_name(point)]["Total"] += count

    return matrix


def add_count_matrices(y0, y1):
    matrix = defaultdict(lambda: defaultdict(int))
    cell_types = set(y0.keys()) | set(y1.keys())
    for cell_type in cell_types:
        areas = set(y0[cell_type].keys()) | set(y1[cell_type].keys())
        for area in areas:
            v0 = y0[cell_type][area]
            v1 = y1[cell_type][area]
            matrix[cell_type][area] = v0 + v1

    return matrix


def export_plots(indices, matrices, keys, cell_types, mri_atlas_dir):
    for k in keys:
        for c in cell_types:
            y = [m[c][k] for m in matrices]
            plt.plot(indices, y)
            plt.savefig(os.path.join(mri_atlas_dir, f"plot_{c}_{k}.png"))
            plt.close()


def main(args):
    excluded = list(map(lambda x: int(x.strip()), read_txt(args.exclude_file)))
    parsed = [parse_one(args, i) if i not in excluded else None
              for i in range(args.start, args.end, args.step)]
    indices, values = zip(*[(i, x) for i, x in zip(range(args.start, args.end, args.step), parsed)
                            if x is not None])
    interp_func = interpolate_dicts(indices, values)
    all_parsed = [interp_func(i) for i in range(min(indices), max(indices), args.step)]
    res = reduce(add_count_matrices, all_parsed)

    pd.DataFrame(res).to_csv(os.path.join(args.mri_atlas_dir, "count_cells.csv"))
    # export plots
    print("Exporting")
    export_plots(indices, values, reduce(lambda x, y: set(x) | set(y), res.values()),
                 res.keys(), args.mri_atlas_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=Path, default=Path("/media/tower/LaCie/REGISTRATION_PROJECT/"
                                                                  "TMP_PIPELINE_M148/config.ini"))
    parser.add_argument("--mri_atlas_dir", type=str, default=None)
    parser.add_argument("--exclude_file", type=str, default=None)
    parser.add_argument("--merged_annotations_mask", type=str, default=None)
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--step", type=int, default=None)
    args_ = fill_with_config(parser)

    main(args_)
