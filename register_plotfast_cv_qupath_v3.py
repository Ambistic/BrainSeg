"""
TODO :
- Load the svg file
- Get the "outline" polygon
- For each svg geometry, create the geojson equivalent (how ?)
- transfer coordinates


To create the json equivalent :
- List all relevant information to keep (more than the geometry) => color (type), point cat
- Build the mapping if required => a starting point is in `prepro_svg.py`
- Create the json with the mapped metadata
"""
from pathlib import Path
import argparse
import numpy as np
from geojson import load, FeatureCollection, dump, utils
import matplotlib
from skimage.registration import phase_cross_correlation
from tqdm import tqdm

matplotlib.use('TkAgg')

from brainseg.geo import svg_to_geojson, simplify_line, simplify_all, transform_geojson, save_geojson
from brainseg.registration import get_affine_transform_matrix
from brainseg.viz.draw import draw_polygon_border, draw_in_mask

import matplotlib.pyplot as plt
import lxml.etree as et

from brainseg.config import fill_with_config
from brainseg.path import build_path_histo
from brainseg.svg.utils import is_polygon, css_to_dict, points_to_numpy
from brainseg.utils import flatten

# this is not good practice
# sys.path.insert(0, str(Path(__file__).parent / "../build/SimpleITK-build/Wrapping/Python/"))

DICT_COLOR_TO_AREA = {
    "rgb(255,0,0)": "outline",
    "rgb(243,0,0)": "claustrum",
    "rgb(228,0,0)": "putamen",
    "rgb(0,248,255)": "white_matter",
    "rgb(0,255,255)": "layer_4",
}


def extract_outline_svg(svg):
    # we assume there is only one outline
    for x in svg.iterchildren():
        if not is_polygon(x):
            continue
        css = x.attrib["style"]
        d = css_to_dict(css)
        if d.get("stroke") == "rgb(255,0,0)":
            p = points_to_numpy(x.attrib["points"])  # TODO need somewhere to shift by 30000, 30000
            print(p.shape, p.min(axis=0), p.max(axis=0))
            return p

    raise IndexError("No outline found in the svg !")


def get_classification_name(obj):
    try:
        name = obj["properties"]["classification"]["name"]
    except (IndexError, KeyError):
        name = None
    return name


def get_outline_mask(geo: FeatureCollection, key: str, shape, downscale):
    size_x, size_y = 0, 0
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
    mask = np.zeros(shape=(int(shape[0] / downscale) + 1, int(shape[1] / downscale) + 1))

    # print(f"Drawing {len(all_coords)} polygons")
    for i, coords in enumerate(all_coords):
        # print(f"Polygon {i} : number of points {len(coords)}")
        p = np.array(coords)
        if p.size == 0:
            continue
        size_x = max(size_x, p.max(axis=0)[0] + 1)
        size_y = max(size_y, p.max(axis=0)[1] + 1)
        if len(coords) > 1000:
            # print("Simplification")
            coords = simplify_line(coords)
            # print(f"After simplification : number of points {len(coords)}")
        draw_in_mask(mask, coords, downscale, value=100)

    return mask, (size_x, size_y)


def merge_geojson(args, json_cv: FeatureCollection, json_fluo: FeatureCollection, transform_matrix):
    # json_cv, json_fluo = json_fluo, json_cv
    final = json_cv.copy()

    def wrapped_transform(x):
        # why the "-" ? I don't know, but it works like this
        x = np.array(list(x) + [args.downscale])  # because it must be 1 at the moment of the transform
        x_ = x / args.downscale
        y_ = x_ @ transform_matrix.T
        y = y_ * args.downscale
        return list(y)

    for feat in json_fluo["features"]:
        obj = utils.map_tuples(wrapped_transform, feat)
        final["features"].append(obj)

    return final


def run(args, slice_id, plotfast_path, cv_path, output):
    with open(cv_path, "r") as f:
        geo_cv = load(f)

    xml = et.parse(plotfast_path)
    svg = xml.getroot()

    geo_fluo = svg_to_geojson(svg, mapping_stroke_classification=DICT_COLOR_TO_AREA)

    geo_fluo = simplify_all(geo_fluo)

    shape = (args.size_x, args.size_y)
    print("Computing cv mask")
    mask_outline_cv, size = get_outline_mask(geo_cv, args.cv_outline_name, shape, args.downscale)
    mat = np.array([
        [1, 0, size[0] / 2],  # here we can integrate a shift if it's not working properly
        [0, 1, size[1] / 2]
    ])
    mat *= 1 / args.mpp_plotfast
    geo_fluo_tmp = transform_geojson(geo_fluo, mat)
    print("Computing fluo mask")
    mask_outline_fluo_tmp, _ = get_outline_mask(geo_fluo_tmp, args.fluo_outline_name, shape, args.downscale)

    # calculate phase correlation
    # is is cv, fluo or fluo, cv ?
    # do we need to inverse x, y afterwards ?
    shift, _, _ = phase_cross_correlation(mask_outline_cv, mask_outline_fluo_tmp)

    # perform shift on geo_fluo transform
    mat = np.array([
        [1, 0, size[0] / 2 + shift[0] * args.downscale],  # here we can integrate a shift if it's not working properly
        [0, 1, size[1] / 2 + shift[1] * args.downscale]
    ])
    mat *= 1 / args.mpp_plotfast
    print(mat, size, shift)
    # return geo_fluo
    geo_fluo = transform_geojson(geo_fluo, mat)

    # recalculate the mask fluo
    mask_outline_fluo, _ = get_outline_mask(geo_fluo, args.fluo_outline_name, shape, args.downscale)

    print("Compute transform function")

    mask_outline_fluo = draw_polygon_border(mask_outline_fluo, border_width=30, polygon_value=100, border_value=200)
    mask_outline_cv = draw_polygon_border(mask_outline_cv, border_width=30, polygon_value=100, border_value=200)
    matrix, res = get_affine_transform_matrix(mask_outline_cv, mask_outline_fluo)
    print("Matrix found", matrix)

    plt.subplot(1, 3, 1)
    plt.imshow(mask_outline_cv)
    plt.subplot(1, 3, 2)
    plt.imshow(mask_outline_fluo)
    plt.subplot(1, 3, 3)
    plt.imshow(res)
    print(f"saving fig at {str(output) + '_qc.png'}")
    plt.savefig(str(output) + "_qc.png")
    plt.close()

    total_geojson = merge_geojson(args, geo_cv, geo_fluo, matrix)

    save_geojson(args, total_geojson, output)
    print(f"saved geojson at {output}")


def main(args):
    for slice_id in tqdm(range(args.start, args.end, args.step)):
        plotfast_path = build_path_histo(args.plotfast_dir, slice_id, args.plotfast_mask)
        cv_path = build_path_histo(args.annotations_dir, slice_id, args.full_annotations_mask)
        output_path = build_path_histo(args.annotations_dir, slice_id, args.merged_annotations_mask)
        try:
            run(args, slice_id, plotfast_path, cv_path, output_path)
        except (FileNotFoundError, OSError) as e:
            print(f"Not found {e}")
        else:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=Path, default=Path("/media/tower/LaCie/REGISTRATION_PROJECT/"
                                                                  "TMP_PIPELINE_M148/config.ini"))
    parser.add_argument("--plotfast_dir", type=Path, default=None)
    parser.add_argument("--plotfast_mask", type=str, default=None)
    parser.add_argument("--annotations_dir", type=str, default=None)
    parser.add_argument("--full_annotations_mask", type=str, default=None)
    parser.add_argument("--merged_annotations_mask", type=str, default=None)
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--step", type=int, default=None)

    parser.add_argument("--fluo-outline-name", default="outline")
    parser.add_argument("--cv-outline-name", default="auto_outline")
    parser.add_argument("-sx", "--size-x", type=int, default=70000)
    parser.add_argument("-sy", "--size-y", type=int, default=40000)
    parser.add_argument("-d", "--downscale", type=int, default=16)
    parser.add_argument("--mpp_plotfast", type=float, default=None)

    args_ = fill_with_config(parser)

    main(args_)
