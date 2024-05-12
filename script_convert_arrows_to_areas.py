#!/usr/bin/env python
# coding: utf-8
import argparse
import traceback

from shapely.geometry import LineString, shape, Point, MultiPolygon, Polygon
from shapely.geometry.collection import GeometryCollection
from shapely.validation import make_valid
import xml.etree.ElementTree as ET
from pathlib import Path
from geojson import load as geo_load, dump as geo_dump
import numpy as np
from brainseg import geo
import pandas as pd

from brainseg.config import fill_with_config
from brainseg.geo import find_feature_by_name, get_closest_point_id, get_closest_point_id_from_list, \
    get_furthest_point_id, find_segment_intersected, find_close_head, valid_tail_point, valid_tail_points
from brainseg.math import distance
from brainseg.parser import get_arrows_from_root, get_arrow_coords
from brainseg.path import build_path_from_mask
from brainseg.utils import filter_index_not_in_list, find_lowest_value_above_threshold, has_larger_range, \
    get_rolling_slice


def check_validity(continuous, broken):
    for i in range(len(continuous) - 1):
        assert continuous[i] < continuous[i + 1], "The continous axis is not valid"
        
    has_broken = False
    for i in range(len(broken) - 1):
        if broken[i] < broken[i + 1]:
            if not has_broken:
                has_broken = True
            else:
                raise ValueError("The broken axis is broken at least twice")
    if has_broken:
        assert broken[-1] > broken[0], f"The broken axis is not consistent, {broken}"


def build_border_from_arrow(arrow, arrows, discard_list, heads):
    tail, head = get_arrow_coords(arrow)[0], get_arrow_coords(arrow)[1]
    mid = []
    while True:
        closest_head_index = find_close_head(tail, heads, threshold=200)
        if closest_head_index is not None:
            discard_list.append(closest_head_index)
            new_arrow = arrows[closest_head_index]
            new_tail, new_head = get_arrow_coords(new_arrow)[0], get_arrow_coords(new_arrow)[1]
            mid.append(tail)
            tail = new_tail
        else:
            break
    return head, mid, tail


def get_tail_intersection(head, mid, tail, wm_coords):
    tail_line = LineString((tail, mid[0] if len(mid) > 0 else head))
    intersection = tail_line.intersection(LineString(wm_coords))
    # Extract intersection point coordinates
    if intersection.geom_type == 'MultiPoint':
        # remove intersection that are not at least twice closer to tail than head
        intersection_points = valid_tail_points(head, tail, intersection.geoms)

        if len(intersection_points) > 0:
            furthest_point_id = get_furthest_point_id(tail, [p.coords[0] for p in intersection_points])
            intersection = intersection_points[furthest_point_id]
        else:
            intersection = None

    elif intersection.geom_type == 'Point':
        if not valid_tail_point(head, tail, intersection):
            intersection = None
    return intersection


def get_closest_with_intersection(head, intersection, tail, wm_coords):
    if isinstance(intersection, Point):  # if exists
        segment = find_segment_intersected(LineString(wm_coords), intersection)
        closest_vertex = segment[0] if distance(head, segment[0]) < distance(head, segment[1]) else segment[1]
        closest_from_tail = np.where((wm_coords == closest_vertex).all(axis=1))[0][0]
    else:
        closest_from_tail = get_closest_point_id(tail, wm_coords)
    return closest_from_tail


def build_df_borders(arrows, heads, wm_coords_list):
    full_arrows = []
    closest_from_heads = []
    closest_from_tails = []
    wm_indexes = []
    discard_list = []
    # find heads
    for arrow in arrows:
        head, mid, tail = build_border_from_arrow(arrow, arrows, discard_list, heads)

        # pick closest location from head at the wm border
        list_id, closest_from_head = get_closest_point_id_from_list(head, wm_coords_list)

        # here check if intersection, if yes pick neighbor of intersecting segment, else nearest point
        wm_coords = wm_coords_list[list_id]
        intersection = get_tail_intersection(head, mid, tail, wm_coords)

        closest_from_tail = get_closest_with_intersection(head, intersection, tail, wm_coords)

        full_arrows.append([head] + mid + [tail])
        closest_from_heads.append(closest_from_head)
        closest_from_tails.append(closest_from_tail)
        wm_indexes.append(list_id)

    df_arrows = pd.DataFrame(dict(
        arrow=filter_index_not_in_list(full_arrows, discard_list),
        head_index=filter_index_not_in_list(closest_from_heads, discard_list),
        tail_index=filter_index_not_in_list(closest_from_tails, discard_list),
        wm_index=filter_index_not_in_list(wm_indexes, discard_list),
    ))
    return df_arrows


def build_polygon_from_sorted_arrows(df, wm_coords):
    """Arrows should be in the head-to-tail format and not in the tail-to-head format"""
    polygons = []
    # head pairs
    prev_row = None
    for i, row in df.iterrows():
        new_polygon_list = []
        if prev_row is None:
            new_polygon_list += get_rolling_slice(wm_coords, row['tail_index'], row['head_index'])
            new_polygon_list += row["arrow"][::]
        else:
            new_polygon_list += prev_row["arrow"][::-1]
            new_polygon_list += get_rolling_slice(wm_coords, prev_row['head_index'], row['head_index'])
            new_polygon_list += row["arrow"][::]
            new_polygon_list += get_rolling_slice(wm_coords, row['tail_index'], prev_row['tail_index'])
        
        new_polygon_list.append(new_polygon_list[0])
        polygons.append(new_polygon_list)
        prev_row = row

    # last arrow
    new_polygon_list = []
    new_polygon_list += prev_row["arrow"][::-1]
    new_polygon_list += get_rolling_slice(wm_coords, prev_row['head_index'], prev_row['tail_index'])
    new_polygon_list.append(new_polygon_list[0])
    polygons.append(new_polygon_list)
    
    return polygons


def make_area_polygons(df_arrows, wm_coords_list):
    all_polygons = []
    for wm_polygon_index in df_arrows["wm_index"].unique():
        wm_coords = list(wm_coords_list[wm_polygon_index])
        df_current = df_arrows[df_arrows["wm_index"] == wm_polygon_index]
        if has_larger_range(df_current["head_index"], df_current["tail_index"]):
            # TBD
            first = find_lowest_value_above_threshold(df_current["head_index"], max(df_current["tail_index"]))
            max_ = max(df_current["head_index"])
            df_current["sort_axis"] = (df_current["head_index"] + max_ - first) % max_
            df_current = df_current.sort_values("sort_axis")
            check_validity(df_current["tail_index"].to_list(), df_current["head_index"].to_list())

        elif has_larger_range(df_current["tail_index"], df_current["head_index"]):
            df_current = df_current.sort_values("head_index")
            check_validity(df_current["head_index"].to_list(), df_current["tail_index"].to_list())

        elif min(df_current["head_index"]) > max(df_current["tail_index"]):
            df_current = df_current.sort_values("head_index")
            check_validity(df_current["head_index"].to_list(), df_current["tail_index"].to_list())

        elif min(df_current["tail_index"]) > max(df_current["head_index"]):
            df_current = df_current.sort_values("head_index")
            check_validity(df_current["head_index"].to_list(), df_current["tail_index"].to_list())

        all_polygons += build_polygon_from_sorted_arrows(df_current, wm_coords)
    return all_polygons


def get_wm(args, geo_feats):
    wm_feature = find_feature_by_name(geo_feats["features"], args.wm_name)
    wm_polygon = shape(wm_feature["geometry"])
    if isinstance(wm_polygon, MultiPolygon):
        wm_polygons = list(wm_polygon.geoms)
    else:
        wm_polygons = [wm_polygon]
    return wm_polygon, wm_polygons


def run(args, segmentation_file, border_file, output_file):
    with open(segmentation_file, "r") as f:
        geo_feats = geo_load(f)

    tree = ET.parse(border_file)
    root = tree.getroot()
    arrows = get_arrows_from_root(root)

    wm_polygon, wm_polygons = get_wm(args, geo_feats)
    wm_coords_list = [np.array(wm.exterior.coords) for wm in wm_polygons]
    heads = [get_arrow_coords(arrow)[1] for arrow in arrows]

    df_arrows = build_df_borders(arrows, heads, wm_coords_list)

    # check state of indexes
    all_polygons = make_area_polygons(df_arrows, wm_coords_list)

    # add the polygons onto the geojson then export it

    for poly in all_polygons:
        p = Polygon(poly)
        if not p.is_valid:
            p = make_valid(p)

        p = p.intersection(wm_polygon)
        if isinstance(p, GeometryCollection):
            p = max(p.geoms, key=lambda x: x.area)
        if isinstance(p, MultiPolygon):
            p = max(p.geoms, key=lambda x: x.area)

        p_array = np.array(p.exterior.coords)
        geo_feats["features"].append(geo.create_qupath_polygon(p_array))

    with open(output_file, "w") as f:
        geo_dump(geo_feats, f)


def main(args):
    for slice_id in range(args.start, args.end, args.step):
        segmentation_file = build_path_from_mask(args.atlas_segmentation_dir, slice_id, args.segmentation_mask)
        border_file = build_path_from_mask(args.atlas_borders_dir, slice_id, args.border_mask)
        output_file = build_path_from_mask(args.atlas_areas_dir, slice_id, args.area_mask)

        if not Path(segmentation_file).exists() or not Path(border_file).exists():
            print("Skip", slice_id)
            continue

        try:
            run(args, segmentation_file, border_file, output_file)
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
    parser.add_argument("--border_mask", type=str, default=None)
    parser.add_argument("--segmentation_mask", type=str, default=None)
    parser.add_argument("--area_mask", type=str, default=None)
    parser.add_argument("--atlas_borders_dir", type=Path, default=None)
    parser.add_argument("--atlas_segmentation_dir", type=Path, default=None)
    parser.add_argument("--atlas_areas_dir", type=Path, default=None)

    parser.add_argument("--wm-name", default="WM")

    args_ = fill_with_config(parser)

    main(args_)
