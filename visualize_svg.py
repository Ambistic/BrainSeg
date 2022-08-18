import argparse
import os
from pathlib import Path

import lxml.etree as ET
from tqdm import tqdm

from brainseg.svg.utils import save_element, center_viewbox, increase_width


def main(args, f):
    xml = ET.parse(str(args.source / f))
    svg = xml.getroot()
    center_viewbox(svg)
    increase_width(svg)

    save_element(svg, str(args.target / f))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source", help="Source folder", type=Path)
    parser.add_argument("-t", "--target", help="Target folder", type=Path)

    args = parser.parse_args()
    for f in tqdm(os.listdir(args.source)):
        if f.endswith(".svg"):
            main(args, f)
