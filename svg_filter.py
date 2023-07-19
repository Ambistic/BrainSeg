import argparse
import os
from pathlib import Path
from shapely.geometry import Polygon
import lxml.etree as ET
from brainseg.svg.utils import is_line, points_to_numpy, save_element


def run_filter_area(f, output, thr):
    xml = ET.parse(f)
    svg = xml.getroot()

    for child in svg.iterchildren():
        if is_line(child):
            d = dict(child.items())
            points = d["points"]

            arr = points_to_numpy(points)
            poly = Polygon(arr)

            if poly.area < thr:
                svg.remove(child)

    save_element(svg, output / f.name)


def main(args):
    for f in os.listdir(args.dir):
        run_filter_area(args.dir / f, args.output, args.threshold)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("-t", "--threshold", type=int)
    parsed_args = parser.parse_args()

    main(parsed_args)
