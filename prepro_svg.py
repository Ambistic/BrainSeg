import lxml.etree as ET
import argparse
from pathlib import Path


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


def css_to_dict(css):
    return dict([map(str.strip, element.split(":")) for element in css.split(";")])


def dict_to_css(d):
    return "; ".join([":".join((k, v)) for k, v in d.items()])


def copy_element(el):
    return ET.fromstring(ET.tostring(el))


def save_element(el, path):
    et = ET.ElementTree(el)
    et.write(str(path), pretty_print=True)


def is_line(node):
    if not isinstance(node.tag, str):
        return False
    
    elif "polygon" in node.tag:
        return True
    
    elif "polyline" in node.tag:
        return True
    
    return False


def remove_noline(svg):
    for x in svg.iterchildren():
        if not is_line(x):
            svg.remove(x)
            

def increase_line_width(svg):
    for x in svg.iterchildren():
        css = x.attrib["style"]
        d = css_to_dict(css)
        if 'stroke-width' in d:
            d["stroke-width"] = "20"
        css = dict_to_css(d)
        x.attrib["style"] = css
        
        
def build_export_with_cond(svg, export_path, keep_lines):
    keep_lines = [DICT_AREA_TO_COLOR[area] for area in keep_lines]
    for x in svg.iterchildren():
        d = css_to_dict(x.attrib["style"])
        if d["stroke"].strip() not in keep_lines:
            svg.remove(x)
            
    save_element(svg, export_path)

        
def build_export_area(svg, filepath):
    # outline only
    build_export_with_cond(
        copy_element(svg),
        filepath.parent / (filepath.stem + "_outline.svg"),
        ["outline"],
    )
    
    # outline + WM
    build_export_with_cond(
        copy_element(svg),
        filepath.parent / (filepath.stem + "_whitematter.svg"),
        ["outline", "white_matter"],
    )
    
    # claustrum
    build_export_with_cond(
        copy_element(svg),
        filepath.parent / (filepath.stem + "_claustrum.svg"),
        ["claustrum"],
    )
    
    # putamen
    build_export_with_cond(
        copy_element(svg),
        filepath.parent / (filepath.stem + "_putamen.svg"),
        ["putamen"],
    )
    
    
def center_viewbox(svg):
    svg.attrib["viewBox"] = "-30000 -30000 60000 60000"
    svg.attrib["height"] = "1200"
    svg.attrib["width"] = "1200"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file")
    
    args = parser.parse_args()
    args.file = Path(args.file)
    
    xml = ET.parse(args.file)
    svg = xml.getroot()
    center_viewbox(svg)
    remove_noline(svg)
    increase_line_width(svg)
    build_export_area(svg, args.file)
    