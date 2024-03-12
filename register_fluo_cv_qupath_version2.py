import sys
from pathlib import Path
import argparse
import numpy as np
from geojson import load, FeatureCollection, dump, utils
from shapely import geometry
import matplotlib.pyplot as plt
from tqdm import tqdm

from brainseg.config import fill_with_config
from brainseg.path import build_path_histo
from brainseg.registration import get_affine_transform_matrix
from brainseg.utils import flatten
from brainseg.viz.draw import draw_in_mask


def get_classification_name(obj):
    try:
        name = obj["properties"]["classification"]["name"]
    except (IndexError, KeyError):
        name = None
    return name


def simplify_line(line, tol=20):
    line = geometry.LineString(np.array(line))
    line = line.simplify(tol)  # arbitrary

    return np.array(line.coords)


def simplify_all(geo: FeatureCollection):
    geo = geo.copy()

    for feat in geo["features"]:
        coords = feat["geometry"]["coordinates"]
        if feat["geometry"]["type"] == "Polygon":
            new_coords = [simplify_line(coords[0]).tolist()]
        elif feat["geometry"]["type"] == "MultiPolygon":
            new_coords = list(map(lambda x: [simplify_line(x[0]).tolist()], coords))
        else:
            new_coords = coords

        feat["geometry"]["coordinates"] = new_coords

    return geo


def get_outline_mask(geo: FeatureCollection, key: str, shape, downscale):
    features = geo["features"]
    all_coords = []
    for feat in features:
        if not get_classification_name(feat) == key:
            continue

        coords = feat["geometry"]["coordinates"]
        if feat["geometry"]["type"] == "Polygon":
            all_coords.append(coords)
        elif feat["geometry"]["type"] == "MultiPolygon":
            all_coords += list(coords)
        else:
            raise ValueError("Not expected to find something else than "
                             "a Polygon or a MultiPolygon")

    all_coords = flatten(all_coords)
    mask = np.zeros(shape=(int(shape[0] / downscale), int(shape[1] / downscale)))

    print(f"Drawing {len(all_coords)} polygons")

    for i, coords in enumerate(all_coords):
        print(f"Polygon {i} : number of points {len(coords)}")
        if len(coords) > 1000:
            print("Simplification")
            coords = simplify_line(coords)
            print(f"After simplification : number of points {len(coords)}")
        draw_in_mask(mask, coords, downscale)

    return mask


def merge_geojson(args, json_cv: FeatureCollection, json_fluo: FeatureCollection, transform_matrix):
    # json_cv, json_fluo = json_fluo, json_cv
    final = json_cv.copy()

    def wrapped_transform(x):
        downscale = args.downscale * args.fluo_scale
        # why the "-" ? I don't know, but it works like this
        x = np.array(x + [downscale])  # because it must be 1 at the moment of the transform
        x_ = x / downscale
        y_ = x_ @ transform_matrix.T
        y = y_ * args.downscale
        return list(y)

    for feat in json_fluo["features"]:
        obj = utils.map_tuples(wrapped_transform, feat)
        final["features"].append(obj)

    return final


def save_geojson(output_path, geo):
    with open(output_path, "w") as f:
        dump(geo, f)


def run(args, slice_id, fluo_path, cv_path, output_path):
    print(cv_path, fluo_path)
    with open(cv_path, "r") as f:
        geo_cv = load(f)

    with open(fluo_path, "r") as f:
        geo_fluo = load(f)

    geo_fluo = simplify_all(geo_fluo)

    shape = (args.size_x, args.size_y)
    print("Computing cv mask")
    mask_outline_cv = get_outline_mask(geo_cv, args.cv_outline_name, shape, args.downscale)
    print("Computing fluo mask")
    mask_outline_fluo = get_outline_mask(geo_fluo, args.fluo_outline_name, shape, args.downscale * args.fluo_scale)

    print("Compute transform function")

    matrix, res = get_affine_transform_matrix(mask_outline_cv, mask_outline_fluo)
    print("Matrix found", matrix)

    if False:
        plt.subplot(1, 3, 1)
        plt.imshow(mask_outline_cv)
        plt.subplot(1, 3, 2)
        plt.imshow(mask_outline_fluo)
        plt.subplot(1, 3, 3)
        plt.imshow(res)
        plt.show()

    total_geojson = merge_geojson(args, geo_cv, geo_fluo, matrix)

    save_geojson(output_path, total_geojson)


def main(args):
    for slice_id in tqdm(range(args.start, args.end, args.step)):
        fluo_path = build_path_histo(args.fluo_dir, slice_id, args.fluo_mask)
        cv_path = build_path_histo(args.annotations_dir, slice_id, args.full_annotations_mask)
        output_path = build_path_histo(args.annotations_dir, slice_id, args.merged_annotations_mask)
        try:
            run(args, slice_id, fluo_path, cv_path, output_path)
        except (FileNotFoundError, OSError) as e:
            print(f"Not found {e}")
        else:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=Path, default=Path("/media/tower/LaCie/REGISTRATION_PROJECT/"
                                                                  "TMP_PIPELINE_M148/config.ini"))

    parser.add_argument("--fluo_dir", type=Path, default=None)
    parser.add_argument("--fluo_mask", type=str, default=None)
    parser.add_argument("--annotations_dir", type=str, default=None)
    parser.add_argument("--full_annotations_mask", type=str, default=None)
    parser.add_argument("--merged_annotations_mask", type=str, default=None)
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--fluo_scale", type=float, default=None)

    parser.add_argument("--fluo-outline-name", default="Contour")
    parser.add_argument("--cv-outline-name", default="auto_outline")
    parser.add_argument("-sx", "--size-x", type=int, default=100000)
    parser.add_argument("-sy", "--size-y", type=int, default=100000)
    parser.add_argument("-d", "--downscale", type=int, default=16)

    args_ = fill_with_config(parser)

    main(args_)
