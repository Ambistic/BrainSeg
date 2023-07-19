import argparse
import os
from pathlib import Path

import numpy as np
from geojson import load, FeatureCollection, dump, utils, loads, dumps
from shapely.ops import transform
from shapely import geometry
from collections.abc import Sequence

from skimage.io import imread

from brainseg.config import fill_with_config
from brainseg.misc.points import transfer_points
from brainseg.path import build_path_histo, build_path_mri
from brainseg.viz.draw import draw_geojson_on_image


def read_histo(file_name):
    if file_name.endswith(".geojson"):
        with open(file_name, "r") as f:
            geo = load(f)
        return geo
    raise ValueError(f"Not recognized extension for {file_name}")


def write_histo(geo, file_name):
    if file_name.endswith(".geojson"):
        with open(file_name, "w") as f:
            dump(geo, f)
    else:
        raise ValueError(f"Not recognized extension for {file_name}")


def geojson_mapping(feature, mapping):
    feature = feature.copy()
    geom = geometry.mapping(
        transform(mapping, geometry.shape(feature["geometry"]))
    )

    feature["geometry"] = loads(dumps(geom))

    return feature


def transform_forward_histo(geo, args):
    geo = geo.copy()
    feats = geo["features"].copy()

    def rescale(x, y):
        return x / args.histo_downscale, y / args.histo_downscale

    new_feats = list(map(
        lambda f: geojson_mapping(f, rescale),
        feats
    ))

    geo["features"] = new_feats
    return geo


def record_point_coordinates(geo):
    geo = geo.copy()
    point_list = []

    def record(x, y):
        if isinstance(x, Sequence):
            raise TypeError("")  # To fail the attempt
        point_list.append((x, y))
        return x, y

    feats = geo["features"].copy()
    list(map(
        lambda f: geojson_mapping(f, record),
        feats
    ))

    return point_list


def transform_from_dict(geo, dict_mapping):
    geo = geo.copy()
    feats = geo["features"].copy()

    def func_mapping(x, y):
        if isinstance(x, Sequence):
            raise TypeError("")  # To fail the attempt
        dx, dy = dict_mapping[(x, y)]
        point = (x + dx, y + dy)
        return point

    new_feats = list(map(
        lambda f: geojson_mapping(f, func_mapping),
        feats
    ))

    geo["features"] = new_feats
    return geo


def transform_forward_histo_mri(geo, args, slide_id):
    transform_path = args.transforms_dir / (str(slide_id).zfill(3) + "_forward") / "TransformParameters.1.txt"
    print(transform_path, transform_path.exists())
    transform_path = str(transform_path)
    point_list = record_point_coordinates(geo)

    transferred_point_list = transfer_points(point_list, transform_path)
    dict_point_transfer = {point: transfer for point, transfer in zip(point_list, transferred_point_list)}
    transformed_geo = transform_from_dict(geo, dict_point_transfer)

    return transformed_geo


def run_slice(args, slice_id):
    histo_file = build_path_histo(args.annotations_dir, slice_id, args.full_annotations_mask)
    histo = read_histo(histo_file)  # geojson handled for now
    histo_resized = transform_forward_histo(histo, args)

    mri_slice_space = transform_forward_histo_mri(histo_resized, args, slice_id)

    output_file = build_path_histo(args.mri_projections_dir, slice_id, args.full_annotations_mask)
    write_histo(mri_slice_space, output_file)

    output_image = str(args.mri_projections_dir / f"annotation_on_mri_{slice_id}.png")
    image = imread(build_path_mri(args.mri_sections_dir, slice_id, "raw"))
    draw_geojson_on_image(image, mri_slice_space, output_image)
    print("end")


def main(args):
    for i in range(args.start, args.end, args.step):
        run_slice(args, i)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=Path, default=Path("/media/tower/LaCie/REGISTRATION_PROJECT/"
                                                                  "TMP_PIPELINE_M148/config.ini"))
    parser.add_argument("--histo_downscale", type=float, default=None)
    parser.add_argument("--full_annotations_mask", type=str, default=None)
    parser.add_argument("--annotations_dir", type=Path, default=None)
    parser.add_argument("--transforms_dir", type=Path, default=None)
    parser.add_argument("--mri_projections_dir", type=Path, default=None)
    parser.add_argument("--mri_sections_dir", type=Path, default=None)
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--step", type=int, default=None)
    args_ = fill_with_config(parser)

    main(args_)
