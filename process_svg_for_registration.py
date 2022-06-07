import lxml.etree as ET
import argparse
from pathlib import Path
import glob
from tqdm import tqdm
import warnings
from functools import lru_cache

from brainseg.svg.parse import get_slice_number, get_neuron_cat
from brainseg.svg.utils import css_to_dict, dict_to_css, copy_element, save_element, center_viewbox, \
    is_point, is_line, increase_width, points_to_numpy, is_polygon

SUFFIX_CELL_FILE = "DY-FB.csv.contour_cells"

DICT_COLOR_TO_AREA = {
    "rgb(255,0,0)": "outline",
    "rgb(0,247,255)": "outline",
    "rgb(243,0,0)": "claustrum",
    "rgb(228,0,0)": "putamen",
    "rgb(0,248,255)": "white_matter",
    "rgb(0,255,255)": "layer_4",
    "rgb(242,0,0)": "claustrum_ventral",
    "rgb(241,0,0)": "claustrum_ache",
    "rgb(227,0,0)": "amygdala",
    "rgb(150,0,255)": "claustrum_limen_insula"
}


@lru_cache
def warn_neuron_cat(cat):
    warnings.warn(f"Category {cat} found but no dye associated !")


def increase_line_width(svg):
    for x in svg.iterchildren():
        css = x.attrib["style"]
        d = css_to_dict(css)
        if 'stroke-width' in d:
            d["stroke-width"] = "100"
        css = dict_to_css(d)
        x.attrib["style"] = css


def build_export_with_cond(svg, export_path, keep_lines, mandatory=False):
    svg = copy_element(svg)
    keep = False
    for x in svg.iterchildren():
        if not is_line(x):
            svg.remove(x)
            continue

        d = css_to_dict(x.attrib["style"])
        if DICT_COLOR_TO_AREA.get(d["stroke"].strip(), "") not in keep_lines:
            svg.remove(x)
        elif points_to_numpy(x.attrib["points"]).shape[0] < args.min_polygon_size and is_polygon(x):
            # print("Removing a too small polygon")
            svg.remove(x)
        else:
            keep = True

    if not (mandatory or keep):
        return

    save_element(svg, export_path)


def neurons_to_str(ls, id_slice):
    return "\n".join(
        [','.join(
            [str(i), x, y, "0", id_slice, dye]
        ) for i, (dye, x, y) in enumerate(ls)]
    )


def create_neuron_file_content(neurons, id_slice):
    str_neurons = neurons_to_str(neurons, id_slice)
    string = "\n".join([
        "CSVF-FILE,0,,,," ,
        "csvf-section-start,header,2,,," ,
        "tag,value,,,," ,
        "Caret-Version,5.64,,,," ,
        "encoding,COMMA_SEPARATED_VALUE_FILE,,,," ,
        "csvf-section-end,header,,,," ,
        "csvf-section-start,Cells,6,,," ,
        "Cell Number,X,Y,Z,Section,Name" ,
        str_neurons,
        "csvf-section-end,Cells,,,,"
    ])
    return string


def run_extract_points(args, svg, export_path, id_slice):
    ls_neurons = []
    last_comment = None
    for x in svg.iterchildren():
        if is_point(x):
            cat = get_neuron_cat(last_comment.text)
            dye = DYE_DICT.get(cat)
            if dye is None:
                warn_neuron_cat(cat)
                continue
            cx = x.attrib["x" if "x" in x.attrib else "cx"]
            cy = x.attrib["y" if "y" in x.attrib else "cy"]
            ls_neurons.append((dye, cx, cy))

        if isinstance(x, ET._Comment):
            last_comment = x
        else:
            last_comment = None

    file_content = create_neuron_file_content(ls_neurons, id_slice)
    with open(export_path + "DY-FB.csv.contour_cells", "w") as f:
        f.write(file_content)


def run_extract_lines(args, svg, export_path):
    build_export_with_cond(svg, export_path + "_pial.svg", ["outline"], mandatory=True)
    build_export_with_cond(svg, export_path + "_white.svg", ["white_matter"], mandatory=True)

    build_export_with_cond(svg, export_path + "_cla.svg", ["claustrum"])
    build_export_with_cond(svg, export_path + "_cla_ven.svg", ["claustrum_ventral"])
    build_export_with_cond(svg, export_path + "_cla_AchE.svg", ["claustrum_ache"])
    build_export_with_cond(svg, export_path + "_cla_li.svg", ["claustrum_limen_insula"])
    build_export_with_cond(svg, export_path + "_pu.svg", ["putamen"])
    build_export_with_cond(svg, export_path + "_amy.svg", ["amygdala"])
    build_export_with_cond(svg, export_path + "_Layer4.svg", ["layer_4"])


def export_vizualize_svg(args, svg, name):
    (args.output / "claustrum").mkdir(parents=True, exist_ok=True)
    build_export_with_cond(svg, args.output / "claustrum" / (name + ".svg"),
                           ["claustrum", "claustrum_ventral", "claustrum_ache",
                            "claustrum_limen_insula"])
    (args.output / "border").mkdir(parents=True, exist_ok=True)
    build_export_with_cond(svg, args.output / "border" / (name + ".svg"),
                           ["outline", "white_matter"])


def main(args, f):
    xml = ET.parse(f)
    svg = xml.getroot()
    id_slice = get_slice_number(Path(f).name)

    center_viewbox(svg)
    increase_width(svg)

    full_name_data = str(args.data / (args.name + str(id_slice)))
    run_extract_points(args, svg, full_name_data, id_slice)
    run_extract_lines(args, svg, full_name_data)

    name = args.name + str(id_slice)
    export_vizualize_svg(args, svg, name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--root", type=Path)
    parser.add_argument("-o", "--output", type=Path)

    parser.add_argument("-n", "--name", help="Name of the monkey")
    parser.add_argument("-e", "--error", action="store_true")
    parser.add_argument("-m", "--min-polygon-size", type=int, default=10)
    parser.add_argument("--dyes", help="Name of the dyes", nargs="+")
    
    args = parser.parse_args()

    args.data = args.output / "data"
    args.data.mkdir(parents=True, exist_ok=True)

    DYE_DICT = {str(i + 1): x for i, x in enumerate(args.dyes)}
    
    for f in tqdm(glob.glob(str(args.root / "**" / "*.svg"), recursive=True)):
        try:
            main(args, Path(f))
        except Exception as e:
            if args.error:
                raise e
            print("An error occurred for", f)
            print("Error was", e)
    