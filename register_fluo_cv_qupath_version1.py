import sys
from pathlib import Path
import argparse
import numpy as np
from scipy.ndimage import affine_transform
from geojson import load, FeatureCollection, dump, utils
from skimage.draw import polygon
from shapely import geometry
import matplotlib.pyplot as plt

from brainseg.utils import flatten

# this is not good practice
sys.path.insert(0, str(Path(__file__).parent / "../build/SimpleITK-build/Wrapping/Python/"))


def get_matrix_from_elastix(fixed, moving):
    import SimpleITK as sitk
    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetFixedImage(sitk.GetImageFromArray(fixed))
    elastixImageFilter.SetMovingImage(sitk.GetImageFromArray(moving))

    parameterMapVector = sitk.VectorOfParameterMap()
    affine_param = sitk.GetDefaultParameterMap("affine")
    affine_param["MaximumNumberOfIterations"] = ['1024']
    affine_param["NumberOfResolutions"] = ['5.000000']
    affine_param["Metric"] = ["AdvancedNormalizedCorrelation"]
    # affine_param["ImageSampler"] = ["Grid"]
    affine_param["NumberOfSpatialSamples"] = ["50000"]

    affine_param["AutomaticTransformInitialization"] = ["true"]
    affine_param["AutomaticTransformInitializationMethod"] = ["CenterOfGravity"]

    parameterMapVector.append(affine_param)
    elastixImageFilter.SetParameterMap(parameterMapVector)
    elastixImageFilter.Execute()

    # second parameter map
    if False:
        param2 = elastixImageFilter.GetTransformParameterMap()[0]
        param2["NumberOfResolutions"] = ['1.000000']
        param2["NumberOfSpatialSamples"] = ["50000"]

        elastixImageFilter.SetParameterMap(param2)
        elastixImageFilter.Execute()

    dict_output = dict(elastixImageFilter.GetTransformParameterMap()[0])
    assert dict_output["Transform"] == ('AffineTransform',)

    sitk.PrintParameterMap(sitk.GetDefaultParameterMap("affine"))
    sitk.PrintParameterMap(affine_param)

    return (
        dict_output["TransformParameters"],
        dict_output["CenterOfRotationPoint"],
        sitk.GetArrayFromImage(elastixImageFilter.GetResultImage())
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
    d, c, b, a, U, V = processed_mat
    cx, cy = map(float, center)

    # u = V - (sy / 2 * b - sx / 2 * (1 - a))
    # v = U - (sx / 2 * c - sy / 2 * (1 - d))

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


def draw_in_mask(mask, poly, downscale):
    """In-place"""
    # print(poly)
    # print(poly[0])
    r, c = map(np.array, zip(*poly))
    r, c = r / downscale, c / downscale
    rr, cc = polygon(r, c)
    mask[rr, cc] = 255


def simplify_line(line, tol=20):
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
    mask = np.zeros(shape=(int(shape[0] / downscale), int(shape[1] / downscale)))

    print(f"Drawing {len(all_coords)} polygons")

    for i, coords in enumerate(all_coords):
        print(f"Polygon {i} : number of points {len(coords)}")
        if len(coords) > 1000:
            print("Simplification")
            coords = simplify_line(coords)
            print(f"After simplification : number of points {len(coords)}")
        draw_in_mask(mask, coords, downscale)

    return mask


def merge_geojson(args, json_cv: FeatureCollection, json_fluo: FeatureCollection, transform_matrix):
    # json_cv, json_fluo = json_fluo, json_cv
    final = json_cv.copy()

    def wrapped_transform(x):
        # why the "-" ? I don't know, but it works like this
        x = np.array(x + [args.downscale])  # because it must be 1 at the moment of the transform
        x_ = x / args.downscale
        y_ = x_ @ transform_matrix.T
        y = y_ * args.downscale
        return list(y)

    for feat in json_fluo["features"]:
        obj = utils.map_tuples(wrapped_transform, feat)
        final["features"].append(obj)

    return final


def save_geojson(args, geo):
    with open(args.output, "w") as f:
        dump(geo, f)


def main(args):
    print(args.cv, args.fluo)
    with open(args.cv, "r") as f:
        geo_cv = load(f)

    with open(args.fluo, "r") as f:
        geo_fluo = load(f)

    geo_fluo = simplify_all(geo_fluo)

    shape = (args.size_x, args.size_y)
    print("Computing cv mask")
    mask_outline_cv = get_outline_mask(geo_cv, args.cv_outline_name, shape, args.downscale)
    print("Computing fluo mask")
    mask_outline_fluo = get_outline_mask(geo_fluo, args.fluo_outline_name, shape, args.downscale)

    print("Compute transform function")

    matrix, res = get_transform_matrix(mask_outline_cv, mask_outline_fluo)
    print("Matrix found", matrix)

    if False:
        plt.subplot(1, 3, 1)
        plt.imshow(mask_outline_cv)
        plt.subplot(1, 3, 2)
        plt.imshow(mask_outline_fluo)
        plt.subplot(1, 3, 3)
        plt.imshow(res)
        plt.show()

    total_geojson = merge_geojson(args, geo_cv, geo_fluo, matrix)

    save_geojson(args, total_geojson)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv", type=Path)  # output is made in the cv
    parser.add_argument("--fluo", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--fluo-outline-name", default="Contour")
    parser.add_argument("--cv-outline-name", default="auto_outline")
    parser.add_argument("-sx", "--size-x", type=int, default=70000)
    parser.add_argument("-sy", "--size-y", type=int, default=35000)
    parser.add_argument("-d", "--downscale", type=int, default=32)

    parsed_args = parser.parse_args()
    main(parsed_args)
