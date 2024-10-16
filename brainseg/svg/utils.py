import numpy as np
from lxml import etree as ET


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


def get_svg_root_plotfast():
    return ET.fromstring(RAW.format(viewbox='viewBox="-30000 -30000 60000 60000"'))


def add_polygon_svg(root, array_points, color):
    poly = ET.Element("polygon", dict(
        points=numpy_to_points(array_points),
        style=f"stroke:{color}; fill:none; stroke-width:1"
    ))
    root.append(poly)


def add_polyline_svg(root, array_points, color):
    poly = ET.Element("polyline", dict(
        points=numpy_to_points(array_points),
        style=f"stroke:{color}; fill:none; stroke-width:1"
    ))
    root.append(poly)


def add_point_svg(root, coordx, coordy, name):
    # maybe the stroke value could be changed
    root.append(ET.Comment(f" Point {name} "))
    stroke = {"stroke-width": "1"}
    point = ET.Element("circle", dict(cx=str(coordx), cy=str(coordy), r="2.500000",
                                      stroke="rgb(0,0,255)", fill="none", **stroke))
    root.append(point)


def css_to_dict(css):
    return dict([map(str.strip, element.split(":")) for element in css.split(";")])


def dict_to_css(d):
    return "; ".join([":".join((k, v)) for k, v in d.items()])


def copy_element(el):
    return ET.fromstring(ET.tostring(el))


def save_element(el, path):
    et = ET.ElementTree(el)
    et.write(str(path), pretty_print=True, xml_declaration=True, standalone='yes')


def is_comment(el):
    if isinstance(el, ET._Comment):
        return True
    return False


def is_point(node):
    if not isinstance(node.tag, str):
        return False

    elif "rect" in node.tag:
        return True

    elif "circle" in node.tag:
        return True

    return False


def is_line(node):
    if not isinstance(node.tag, str):
        return False

    elif "polygon" in node.tag:
        return True

    elif "polyline" in node.tag:
        return True

    return False


def is_polygon(node):
    if not isinstance(node.tag, str):
        return False

    elif "polygon" in node.tag:
        return True

    return False


def is_point(node):
    if not isinstance(node.tag, str):
        return False

    elif "rect" in node.tag:
        return True

    elif "circle" in node.tag:
        return True

    return False


def remove_noline(svg):
    for x in svg.iterchildren():
        if not is_line(x):
            svg.remove(x)


def numpy_to_points(arr):
    arr = arr.astype(int).astype(str)
    return " ".join([",".join(x) for x in arr])


def points_to_numpy(points):
    ls = points.split(" ")
    ls = filter(lambda x: len(x) > 0, ls)
    ls = list(map(lambda x: x.split(","), ls))
    arr = np.array(ls).astype(float)

    return arr


def center_viewbox(svg):
    ls_points = []
    for x in svg.iterchildren():
        if not is_line(x):
            continue

        ls_points.append(points_to_numpy(x.attrib["points"]))

    all_points = np.vstack(ls_points)

    min_x, min_y = np.min(all_points, axis=0).astype(int)

    svg.attrib["viewBox"] = f"{min_x - 500} {min_y - 500} 60000 60000"
    svg.attrib["width"] = "1000"
    svg.attrib["height"] = "1000"


def increase_width(svg, width=100):
    for x in svg.iterchildren():
        if not is_line(x):
            if is_point(x):
                x.attrib["stroke-width"] = str(width)
            continue

        style = css_to_dict(x.attrib["style"])
        style["stroke-width"] = str(width)
        x.attrib["style"] = dict_to_css(style)
