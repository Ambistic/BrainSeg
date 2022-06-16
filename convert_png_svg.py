from pathlib import Path

import aicspylibczi
import numpy as np
import potrace
import argparse
import lxml.etree as ET

from PIL import Image
from scipy import ndimage

from brainseg.svg.utils import save_element, is_line, points_to_numpy, numpy_to_points

RAW = r"""<?xml version="1.0" standalone="yes"?>
<!-- Generator by matlab -->
<svg
xmlns="http://www.w3.org/2000/svg"
xmlns:xlink="http://www.w3.org/1999/xlink"
xmlns:svg="http://www.w3.org/2000/svg"
Version="1.1"
viewBox="-30000 -30000 60000 60000"
preserveAspectRatio="xMidYMid meet">
</svg>
"""


def get_size(slide_name):
    slide = aicspylibczi.CziFile(slide_name)
    bbox = slide.get_mosaic_bounding_box()

    return np.array([bbox.w, bbox.h])


def scale_contour(args, contour):
    contour = contour * args.downscale * args.mpp
    return contour


def get_contours(args):
    mask = np.asarray(Image.open(args.file))
    mask = ndimage.gaussian_filter(mask, sigma=(5, 5), order=0)
    bmp = potrace.Bitmap(mask > args.threshold)
    path = bmp.trace(alphamax=0.0, turnpolicy=potrace.TURNPOLICY_BLACK, turdsize=args.min_surface)
    return [c.tesselate() for c in path]


def strip_svg(root):
    for ch in root.iterchildren():
        if not is_line(ch):
            continue
        # remove unnecessary stuff
        ch.attrib["points"] = numpy_to_points(points_to_numpy(ch.attrib["points"]))


def save_svg(args, contours):
    if args.reference is None:
        root = ET.fromstring(RAW)
    else:
        xml = ET.parse(args.reference)
        root = xml.getroot()
        strip_svg(root)
    for contour in contours:
        poly = ET.Element("polyline", dict(
            points=numpy_to_points(contour),
            style="stroke:rgb(0,248,255); fill:none; stroke-width:100"
        ))
        root.append(poly)
    save_element(root, str(args.file.parent / args.file.stem) + ".svg")


def main(args):
    contours = get_contours(args)
    contours = map(lambda x: scale_contour(args, x), contours)
    slide_size = get_size(args.slide) / 2 if args.slide is not None else np.array([0, 0])
    contours = map(lambda x: x - slide_size, contours)
    save_svg(args, contours)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-f", "--file")
    parser.add_argument("-r", "--reference", help="Reference svg file", default=None)
    parser.add_argument("--slide", help="Reference slide file for translation", default=None)
    parser.add_argument("-d", "--downscale", type=int)
    parser.add_argument("--mpp", type=float, default=0.88)
    parser.add_argument("-s", "--min-surface", type=int, default=100)
    parser.add_argument("-t", "--threshold", type=int, default=100)

    args = parser.parse_args()
    args.file = Path(args.file)

    main(args)
