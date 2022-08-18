import lxml.etree as ET
import argparse
from pathlib import Path
import os
import glob
from PIL import Image
import numpy as np
from tqdm import tqdm

from brainseg.svg.utils import css_to_dict, dict_to_css, copy_element, save_element, remove_noline

DICT_COLOR_TO_AREA = {
    "rgb(255,0,0)": "outline",
    "rgb(243,0,0)": "claustrum",
    "rgb(228,0,0)": "putamen",
    "rgb(0,248,255)": "white_matter",
    "rgb(0,255,255)": "layer_4",
}

DICT_AREA_TO_COLOR = {
    v: k for k, v in DICT_COLOR_TO_AREA.items()
}


def increase_line_width(svg):
    for x in svg.iterchildren():
        css = x.attrib["style"]
        d = css_to_dict(css)
        if 'stroke-width' in d:
            d["stroke-width"] = "20"
        css = dict_to_css(d)
        x.attrib["style"] = css


def convert_to_png(source, target):
    raw_command = "convert -density 400.2 -colorspace Gray '{source}' '{target}'"
    command = raw_command.format(source=source, target=target)
    os.system(command)


def binarize(f):
    img = Image.open(f)
    arr = np.asarray(img)
    arr2 = arr * (arr == 255)
    n_img = Image.fromarray(arr2)
    n_img.save(f)


def build_export_with_cond(svg, export_path, keep_lines):
    keep_lines = [DICT_AREA_TO_COLOR[area] for area in keep_lines]
    for x in svg.iterchildren():
        d = css_to_dict(x.attrib["style"])
        if d["stroke"].strip() not in keep_lines:
            svg.remove(x)
            
    export_path.parent.mkdir(exist_ok=True, parents=True)
    save_element(svg, export_path)
    export_png_path = export_path.parent / (export_path.stem + ".png")
    convert_to_png(export_path, export_png_path)
    binarize(export_png_path)
        
        
def build_export_area(svg, filepath, datafolder):
    # outline only
    build_export_with_cond(
        copy_element(svg),
        datafolder / filepath.stem / "outline.svg",
        ["outline"],
    )
    
    # outline + WM
    build_export_with_cond(
        copy_element(svg),
        datafolder / filepath.stem / "whitematter.svg",
        ["outline", "white_matter"],
    )
    
    # claustrum
    build_export_with_cond(
        copy_element(svg),
        datafolder / filepath.stem / "claustrum.svg",
        ["claustrum"],
    )
    
    # putamen
    build_export_with_cond(
        copy_element(svg),
        datafolder / filepath.stem / "putamen.svg",
        ["putamen"],
    )


def center_viewbox(svg):
    svg.attrib["viewBox"] = "-30000 -30000 60000 60000"
    svg.attrib["height"] = "1200"
    svg.attrib["width"] = "1200"


def main(args, f):
    xml = ET.parse(f)
    svg = xml.getroot()
    center_viewbox(svg)
    remove_noline(svg)
    increase_line_width(svg)
    build_export_area(svg, f, args.data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--root")
    parser.add_argument("-d", "--data")
    
    args = parser.parse_args()
    args.data = Path(args.data)
    args.root = Path(args.root)
    
    for f in tqdm(glob.glob(str(args.root / "**" / "*.svg"), recursive=True)):
        try:
            main(args, Path(f))
        except Exception as e:
            print("An error occurred for", f)
            print("Error was", e)
    