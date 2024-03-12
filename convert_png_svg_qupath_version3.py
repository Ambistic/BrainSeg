"""TODO
- implement the multipolygon with holes
- manage multi output with a list, if the list is empty it means "all"
"""

from pathlib import Path

import aicspylibczi
import numpy as np
import potrace
import argparse
import lxml.etree as ET
from shapely import geometry
from shapely.geometry import Polygon as ShapelyPolygon, MultiPolygon as ShapelyMultiPolygon
from geojson import FeatureCollection, LineString, Polygon, Feature, dump

from PIL import Image
from scipy import ndimage

from brainseg.config import fill_with_config
from brainseg.geo import quickfix_multipolygon
from brainseg.path import build_path_histo_segmentation
from brainseg.polygon import build_polygon_correspondences
from brainseg.svg.utils import save_element, is_line, points_to_numpy, numpy_to_points
from brainseg.utils import getIfromRGB

RAW = r"""<?xml version="1.0" standalone="yes"?>
<!-- Generator by matlab -->
<svg
xmlns="http://www.w3.org/2000/svg"
xmlns:xlink="http://www.w3.org/1999/xlink"
xmlns:svg="http://www.w3.org/2000/svg"
Version="1.1"
{viewbox}
preserveAspectRatio="xMidYMid meet">
</svg>
"""

SIGMA = 15
CURVE_TOL = 10
UPSAMPLING = 2


def get_size(slide_name):
    slide = aicspylibczi.CziFile(slide_name)
    bbox = slide.get_mosaic_bounding_box()

    return np.array([bbox.w, bbox.h])


def scale_contour(args, contour):
    contour = contour * args.downscale * args.mpp
    return contour


def get_contours(args):
    mask = np.asarray(Image.open(args.wm_file))
    bmp = potrace.Bitmap(mask > args.threshold)
    path_wm = bmp.trace(alphamax=0.0, turnpolicy=potrace.TURNPOLICY_BLACK, turdsize=args.min_surface)

    mask = np.asarray(Image.open(args.outline_file))
    bmp = potrace.Bitmap(mask > args.threshold)
    path_outline = bmp.trace(alphamax=0.0, turnpolicy=potrace.TURNPOLICY_BLACK, turdsize=args.min_surface)

    return [c.tesselate() for c in path_wm], [c.tesselate() for c in path_outline]


def get_contours_v2(args, slice_id):
    wm_file = build_path_histo_segmentation(
            args.annotations_dir, slice_id, args.annotations_mask, "gm", "mask"
        )
    outline_file = build_path_histo_segmentation(
            args.annotations_dir, slice_id, args.annotations_mask, "pial", "mask"
        )
    mask = np.asarray(Image.open(wm_file))
    mask = ndimage.zoom(mask, UPSAMPLING, order=3)
    mask = ndimage.gaussian_filter(mask, sigma=(SIGMA, SIGMA), order=0)
    mask = mask > args.threshold
    # why ? it's too much I think
    # mask = ndimage.binary_erosion(mask, iterations=SIGMA)
    bmp = potrace.Bitmap(mask)
    path_wm = bmp.trace(alphamax=0.0, turnpolicy=potrace.TURNPOLICY_BLACK, turdsize=args.min_surface)

    mask = np.asarray(Image.open(outline_file))
    mask = ndimage.zoom(mask, UPSAMPLING, order=3)
    mask = ndimage.gaussian_filter(mask, sigma=(SIGMA, SIGMA), order=0)
    mask = mask > args.threshold
    # why ?
    # mask = ndimage.binary_erosion(mask, iterations=SIGMA)
    bmp = potrace.Bitmap(mask)
    path_outline = bmp.trace(alphamax=0.0, turnpolicy=potrace.TURNPOLICY_BLACK, turdsize=args.min_surface)

    return [c.tesselate() / UPSAMPLING for c in path_wm], [c.tesselate() / UPSAMPLING for c in path_outline]


def get_non_outline(args):
    mask = np.asarray(Image.open(args.outline_file))
    mask = ndimage.gaussian_filter(mask, sigma=(SIGMA, SIGMA), order=0)
    binary_mask = mask < args.threshold  # !! lower because it's "non"
    kernel = np.array([
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0],
    ])
    non_outline = ndimage.binary_dilation(binary_mask, structure=kernel, iterations=15)

    return non_outline


def filter_point(args, x, y, non_outline):
    # y, x = int(x / args.downscale), int(y / args.downscale)
    y, x = int(x), int(y)
    if (x < 0) or (y < 0) or (x >= non_outline.shape[0]) or (y >= non_outline.shape[1]):
        return False
    return not non_outline[x, y]


def filter_line(args, line, non_outline):
    return list(filter(
        lambda x: filter_point(args, x[0], x[1], non_outline),
        line
    ))


def simplify_line(line):
    line = geometry.LineString(np.array(line))
    line = line.simplify(CURVE_TOL)  # arbitrary

    return np.array(line.coords)


def strip_svg(root):
    for ch in root.iterchildren():
        if not is_line(ch):
            continue
        # remove unnecessary stuff
        ch.attrib["points"] = numpy_to_points(points_to_numpy(ch.attrib["points"]))


def save_svg(args, contours_wm, contours_outline, slice_id):
    wm_file = build_path_histo_segmentation(
        args.annotations_dir, slice_id, args.annotations_mask, "gm", "mask"
    )
    if args.viewbox == "illustrator":
        size = Image.open(wm_file).size
        # TODO correct the size and add a width + height
        scaled0, scaled1 = size[0] / args.mpp, size[1] / args.mpp
        # other = f'width="{scaled0 / 7.4 / 2 * 1.01}" height="{scaled1 / 7.4 / 2 * 1.01}"'
        other = f'width="{scaled0 / 7.4 * args.downscale / 32}" ' \
                f'height="{scaled1 / 7.4 * args.downscale / 32}"'
        viewbox = f'viewBox="0 0 {scaled0 * args.downscale / 32 / 1.01} {scaled1 * args.downscale / 32 / 1.01}" {other}'
        root = ET.fromstring(RAW.format(viewbox=viewbox))
    elif args.viewbox == "default":
        size = Image.open(wm_file).size
        # TODO correct the size and add a width + height
        scaled0, scaled1 = size[0] / args.mpp, size[1] / args.mpp
        # other = f'width="{scaled0 / 7.4 / 2 * 1.01}" height="{scaled1 / 7.4 / 2 * 1.01}"'
        other = f'width="{scaled0 / 7.4 * args.downscale / 8}" ' \
                f'height="{scaled1 / 7.4 * args.downscale / 8}"'
        viewbox = f'viewBox="0 0 {scaled0 * args.downscale} {scaled1 * args.downscale}" {other}'
        root = ET.fromstring(RAW.format(viewbox=viewbox))
    elif args.viewbox == "plotfast":
        root = ET.fromstring(RAW.format(viewbox='viewBox="-30000 -30000 60000 60000"'))
    else:
        raise ValueError(f"Viewbox parameter is not recognized {args.viewbox}")

    for contour in contours_wm:
        poly = ET.Element("polyline", dict(
            points=numpy_to_points(contour),
            style=f"stroke:rgb(0,248,255); fill:none; stroke-width:{args.stroke_width}"
        ))
        root.append(poly)

    for contour in contours_outline:
        poly = ET.Element("polygon", dict(
            points=numpy_to_points(contour),
            style=f"stroke:rgb(255,0,0); fill:none; stroke-width:{args.stroke_width}"
        ))
        root.append(poly)

    output = build_path_histo_segmentation(
            args.annotations_dir, slice_id, args.annotations_mask, "seg", "svg"
        )
    save_element(root, output)


def save_geojson(args, contours_wm, contours_outline, slice_id):
    all_objects = []

    polys_wm, inners_wm = build_polygon_correspondences(contours_wm)
    s_ps = [ShapelyPolygon(x, holes=inners_wm[i]) for i, x in enumerate(polys_wm)]
    multi_poly = ShapelyMultiPolygon(s_ps)
    geom = multi_poly.__geo_interface__

    feat = Feature(geometry=geom, properties=dict(
        object_type="annotation",
        classification={"name": "auto_wm", "colorRGB": getIfromRGB((0, 0, 255))}
    ))
    all_objects.append(feat)

    polys_outline, inners_outline = build_polygon_correspondences(contours_outline)
    s_ps = [ShapelyPolygon(x, holes=inners_outline[i]) for i, x in enumerate(polys_outline)]
    multi_poly = ShapelyMultiPolygon(s_ps)
    geom = multi_poly.__geo_interface__

    feat = Feature(geometry=geom, properties=dict(
        object_type="annotation",
        classification={"name": "auto_outline", "colorRGB": getIfromRGB((255, 0, 0))}
    ))
    all_objects.append(feat)

    geo_object = quickfix_multipolygon(FeatureCollection(all_objects))
    output = build_path_histo_segmentation(
        args.annotations_dir, slice_id, args.annotations_mask, "seg", "geojson"
    )
    with open(output, "w") as f:
        dump(geo_object, f)


def run(args, slice_id):
    # get contours
    contours_wm, contours_outline = get_contours_v2(args, slice_id)

    # this must be used only in case of the need of having a line
    if args.wm_lines:
        non_outline = get_non_outline(args)
        contours_wm = map(lambda x: filter_line(args, x, non_outline), contours_wm)

    contours_wm = map(lambda x: scale_contour(args, np.array(x)), contours_wm)
    contours_outline = map(lambda x: scale_contour(args, np.array(x)), contours_outline)

    contours_wm = filter(lambda x: len(x) >= 2, contours_wm)  # filter empty arrays
    contours_outline = filter(lambda x: len(x) >= 3, contours_outline)  # filter empty arrays

    contours_wm = map(simplify_line, contours_wm)
    contours_outline = map(simplify_line, contours_outline)

    contours_wm = map(np.array, contours_wm)
    contours_outline = map(np.array, contours_outline)

    contours_wm = list(contours_wm)
    contours_outline = list(contours_outline)

    save_svg(args, contours_wm, contours_outline, slice_id)
    save_geojson(args, contours_wm, contours_outline, slice_id)


def main(args):
    for slice_id in range(args.start, args.end, args.step):
        try:
            run(args, slice_id)
        except FileNotFoundError:
            print(f"File not found for {slice_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--config", type=Path, default=Path("/media/tower/LaCie/REGISTRATION_PROJECT/"
                                                                  "TMP_PIPELINE_M148/config.ini"))
    parser.add_argument("--annotations_dir", type=str, default=None)
    parser.add_argument("--annotations_mask", type=str, default=None)
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--step", type=int, default=None)

    parser.add_argument("-d", "--downscale", type=int, default=8)
    parser.add_argument("--mpp", type=float, default=1)  # before it was `0.88` but I don't know why
    parser.add_argument("-s", "--min-surface", type=int, default=10000)
    parser.add_argument("-t", "--threshold", type=int, default=150)
    parser.add_argument("-v", "--viewbox", default="default")
    parser.add_argument("--stroke-width", type=int, default=100)
    parser.add_argument("--wm_lines", action="store_true")

    args_ = fill_with_config(parser)

    main(args_)
