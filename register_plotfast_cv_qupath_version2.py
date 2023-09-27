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
import tempfile
from pathlib import Path
import argparse
import itk
import re
import numpy as np
from scipy.ndimage import affine_transform, binary_erosion
from geojson import load, FeatureCollection, dump, utils, Point, Polygon, Feature
from scipy.ndimage import binary_dilation
from skimage.draw import polygon
from shapely import geometry
import matplotlib
from tqdm import tqdm

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import lxml.etree as et

from brainseg.config import fill_with_config
from brainseg.path import build_path_histo
from brainseg.svg.utils import is_polygon, css_to_dict, points_to_numpy, is_point, is_comment
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


def draw_polygon_border(image, border_width=10, polygon_value=255, border_value=100):
    # Create a structuring element for dilation
    structuring_element = np.ones((border_width, border_width), dtype=np.uint8)

    # Perform dilation on the polygons
    dilated = binary_dilation(image == polygon_value, structure=structuring_element)
    eroded = binary_erosion(image == polygon_value, structure=structuring_element)

    # Create the border by subtracting the original image from the dilated image
    # border = dilated & (image != polygon_value)
    border = dilated & ~eroded

    # Assign the border value to the border pixels
    result = np.where(border, border_value, image)

    return result


def get_point_name_from_comment(comment):
    match = re.search(r'<!-- Point (.*?) -->', comment)
    if match:
        s = match.group(1)
        return s
    return None


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


def svg_to_geojson(svg, mapping_stroke_classification):
    features = []
    last_comment = None
    for x in svg.iterchildren():
        if is_point(x):
            point_name = get_point_name_from_comment(str(last_comment))
            if "x" in x.attrib:
                kx, ky = "x", "y"
            else:
                kx, ky = "cx", "cy"
            point = Point((float(x.attrib[kx]), float(x.attrib[ky])))
            properties = {"classification": {"name": point_name}}
            feat = Feature(geometry=point, properties=properties)
            features.append(feat)

        if is_polygon(x):
            # process color
            # add a polygon
            coords = points_to_numpy(x.attrib["points"])
            polygon = Polygon([coords.tolist()])
            style = css_to_dict(x.attrib["style"])
            category = mapping_stroke_classification.get(style["stroke"], "none")
            if category == "none":
                print(style["stroke"])
            properties = {"classification": {"name": category}}
            feat = Feature(geometry=polygon, properties=properties)
            features.append(feat)

        if is_comment(x):
            last_comment = x
        else:
            last_comment = None

    return FeatureCollection(features)


def get_matrix_from_elastix(image_histo, image_mri):
    fixed = itk.GetImageFromArray(image_histo.astype(np.float32), is_vector=False)
    moving = itk.GetImageFromArray(image_mri.astype(np.float32), is_vector=False)

    parameter_object = itk.ParameterObject.New()
    parameter_map_affine = parameter_object.GetDefaultParameterMap('affine')

    parameter_map_affine["Transform"] = ['EulerTransform']
    parameter_map_affine["MaximumNumberOfIterations"] = ['1024']
    parameter_map_affine["NumberOfResolutions"] = ['6.000000']  # could be 5
    parameter_map_affine["Metric"] = ["AdvancedNormalizedCorrelation"]
    parameter_map_affine["NumberOfSpatialSamples"] = ["50000"]

    # parameter_map_affine["AutomaticTransformInitialization"] = ["true"]
    # parameter_map_affine["AutomaticTransformInitializationMethod"] = ["CenterOfGravity"]

    parameter_object.AddParameterMap(parameter_map_affine)

    # Call registration function and specify output directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        result_image, result_transform_parameters = itk.elastix_registration_method(
            fixed, moving,
            parameter_object=parameter_object,
            output_directory=str(tmp_dir))  # TODO replace with a tmp

    return (
        result_transform_parameters.GetParameter(0, "TransformParameters"),
        result_transform_parameters.GetParameter(0, "CenterOfRotationPoint"),
        np.array(result_image),
    )


def get_transform_matrix(fixed, moving):
    """
    Fixed and moving such that moving = fixed @ affine_matrix
    In other words, the moving is the result of the transformation
    of the fixed image with the returned matrix
    """
    assert isinstance(fixed, np.ndarray)
    assert isinstance(moving, np.ndarray)
    assert fixed.shape == moving.shape

    sx, sy = fixed.shape
    mat, center, res = get_matrix_from_elastix(moving, fixed)  # maybe the opposite
    processed_mat = map(float, mat)
    # s, theta, U, V = processed_mat
    theta, U, V = processed_mat
    cx, cy = map(float, center)

    # u = V - (sy / 2 * b - sx / 2 * (1 - a))
    # v = U - (sx / 2 * c - sy / 2 * (1 - d))

    # print(f"\n ==== S = {s} ==== \n")

    a, b, c, d = np.cos(theta), np.sin(theta), -np.sin(theta), np.cos(theta)
    u = V - (cx * b - cy * (1 - a))
    v = U - (cy * c - cx * (1 - d))

    return np.array([[a, b, u], [c, d, v]]), res


def get_transform_function(fixed, moving):
    affine_matrix = get_transform_matrix(fixed, moving)
    return lambda x: affine_transform(x, affine_matrix)


def get_classification_name(obj):
    try:
        name = obj["properties"]["classification"]["name"]
    except (IndexError, KeyError):
        name = None
    return name


def draw_in_mask(mask, poly, downscale, value=255):
    """In-place"""
    # print(poly)
    # print(poly[0])
    r, c = map(np.array, zip(*poly))
    r, c = r / downscale, c / downscale
    rr, cc = polygon(r, c)

    # filter
    valid_indices = (rr < mask.shape[0]) & (cc < mask.shape[1])
    rr = rr[valid_indices]
    cc = cc[valid_indices]

    mask[rr, cc] = value


def simplify_line(line, tol=20):
    if len(line) < 3:
        return np.array(line)

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


def transform_geojson(json_obj, transform_matrix):
    json_obj = json_obj.copy()

    def wrapped_transform(x):
        # why the "-" ? I don't know, but it works like this
        x = np.array(x + [1])  # because it must be 1 at the moment of the transform
        y_ = x @ transform_matrix.T
        return list(y_)

    ls = []

    for feat in json_obj["features"]:
        obj = utils.map_tuples(wrapped_transform, feat)
        ls.append(obj)

    json_obj["features"] = ls

    return json_obj


def save_geojson(args, geo, output):
    with open(output, "w") as f:
        dump(geo, f)


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
    geo_fluo = transform_geojson(geo_fluo, mat)
    print("Computing fluo mask")
    mask_outline_fluo, _ = get_outline_mask(geo_fluo, args.fluo_outline_name, shape, args.downscale)

    print("Compute transform function")

    mask_outline_fluo = draw_polygon_border(mask_outline_fluo, border_width=30, polygon_value=100, border_value=200)
    mask_outline_cv = draw_polygon_border(mask_outline_cv, border_width=30, polygon_value=100, border_value=200)
    matrix, res = get_transform_matrix(mask_outline_cv, mask_outline_fluo)
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
    parser.add_argument("-d", "--downscale", type=int, default=32)
    parser.add_argument("--mpp_plotfast", type=float, default=None)

    args_ = fill_with_config(parser)

    main(args_)
