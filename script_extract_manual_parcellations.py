import argparse
from pathlib import Path

import geojson
import geopandas as gpd
import os
from typing import Dict
from shapely.geometry import GeometryCollection, Polygon, MultiPolygon, MultiPoint, Point
import pandas as pd
from collections import defaultdict
from itertools import product

from brainseg.config import fill_with_config
from brainseg.path import build_path_histo

import xlwt
from tqdm import tqdm


class DuplicateError(Exception):
    pass


class PolygonWithoutClassification(Exception):
    pass


class UnknownNeuronLabel(ValueError):
    pass


def _get_classification_name(row):
    if isinstance(row["classification"], dict):
        return row["classification"].get("name")
    else:
        raise PolygonWithoutClassification()


def count_points_inside(points, poly):
    if isinstance(points, Point):
        return 1 if points.within(poly) else 0
    points_inside = [point.within(poly) for point in points.geoms]
    count_inside = sum(points_inside)
    return count_inside


def extract_neurons(geo) -> Dict[str, GeometryCollection]:
    neurons = defaultdict(list)

    for index, row in geo.iterrows():
        if row["objectType"] != "annotation":
            continue

        geometry = row['geometry']

        if isinstance(geometry, (Point, MultiPoint)):
            class_name = row["name"]

            neurons[class_name].append(row["geometry"])
    return neurons


def extract_areas(geo) -> Dict[str, GeometryCollection]:
    areas = defaultdict(list)

    for index, row in geo.iterrows():
        if row["objectType"] != "annotation":
            continue

        geometry = row['geometry']

        if isinstance(geometry, (Polygon, MultiPolygon)):
            class_name = _get_classification_name(row)

            areas[class_name].append(row["geometry"])
    return areas


def count_neurons(areas, neurons) -> pd.DataFrame:
    counts = defaultdict(dict)
    for (neuron_class, neuron_geoms), (area_class, area_geoms) in product(neurons.items(), areas.items()):
        counts[neuron_class][area_class] = sum([
            count_points_inside(neuron_geom, area_geom)
            for neuron_geom in neuron_geoms
            for area_geom in area_geoms
        ])

    return pd.DataFrame(counts)


def export_counts(counts, metadata):
    df = counts.copy()
    df.rename(columns=metadata, inplace=True)

    # Ensure the existence of specified columns and fill them with 0s if they don't exist
    for column in ["Cat 1", "Cat 2", "Cat 3", "Cat 4", "Cat 5"]:
        if column not in df.columns:
            df[column] = 0

    return df


def run(geopath, metadata):
    geo = gpd.read_file(geopath)
    areas = extract_areas(geo)
    neurons = extract_neurons(geo)
    # print(areas, neurons)
    df_count = count_neurons(areas, neurons)
    df_formatted = export_counts(df_count, metadata)
    extra_cols = set(df_formatted.columns) - set(["Cat 1", "Cat 2", "Cat 3", "Cat 4", "Cat 5"])

    if extra_cols:
        raise UnknownNeuronLabel(list(extra_cols))

    df_formatted.rename(index=dict(Contour="Total"), inplace=True)
    df_formatted.loc["Contour"] = 0
    if "Total" not in df_formatted.index:
        df_formatted.loc["Total"] = 0

    df_formatted.to_excel(geopath + ".xlsx", index_label="Region")


def main(args):
    metadata_list = map(str.strip, args.marker_types.split(","))
    metadata = {x: f"Cat {i + 1}" for i, x in enumerate(metadata_list)}
    print(f"Metadata : {metadata}")

    for slice_id in tqdm(range(args.start, args.end, args.step)):
        geopath = build_path_histo(args.annotations_dir, slice_id, args.manual_annotations_mask)
        if os.path.exists(geopath):
            try:
                run(geopath, metadata)
            except Exception as e:
                print("Error with", slice_id, type(e), e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=Path, default=None)
    parser.add_argument("--marker_types", type=str, default=None)
    parser.add_argument("--annotations_dir", type=str, default=None)
    parser.add_argument("--manual_annotations_mask", type=str, default=None)
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--step", type=int, default=None)
    args_ = fill_with_config(parser)

    main(args_)
