import argparse
from pathlib import Path

import geojson
import numpy as np
from geojson import FeatureCollection
from scipy.ndimage import gaussian_filter
from shapely import geometry

from shapely.validation import make_valid
from skimage.io import imread
from skimage.measure import find_contours
from skimage.segmentation import expand_labels

from brainseg.config import fill_with_config
from brainseg.geo import quickfix_multipolygon, quickfix_multipolygon_shapely, transform_backward_histo, \
    transform_rescale, record_point_coordinates, transform_from_dict
from brainseg.misc.points import transfer_points
from brainseg.path import build_path_histo, build_path_mri
from brainseg.polygon import build_polygon_correspondences
from shapely.geometry import Polygon as ShapelyPolygon, MultiPolygon as ShapelyMultiPolygon, shape

from brainseg.utils import getIfromRGB, get_processing_type, read_histo, write_histo, read_atlas, read_txt
from brainseg.viz.draw import draw_geojson_on_image, draw_polygon


def transform_backward_mri_histo(args, geo, slide_id, hemisphere="both"):
    if hemisphere == "both":
        folder = "."
    elif hemisphere == "right":
        folder = "right"
    elif hemisphere == "left":
        folder = "left"
    else:
        raise ValueError(f"Unrecognized `hemisphere` value : {hemisphere}")

    transform_path = args.transforms_dir / (str(slide_id).zfill(3) + "_backward") / folder / "TransformParameters.1.txt"
    if not transform_path.exists():
        raise FileExistsError(f"Transform for {slide_id} has not been computed")

    transform_path = str(transform_path)
    point_list = record_point_coordinates(geo)

    transferred_point_list = transfer_points(point_list, transform_path)
    dict_point_transfer = {point: transfer for point, transfer in zip(point_list, transferred_point_list)}
    transformed_geo = transform_from_dict(geo, dict_point_transfer)

    return transformed_geo


def smooth_atlas(atlas, smooth_value=3):
    cat = np.unique(atlas.flatten())
    matrix = np.zeros((atlas.shape[0], atlas.shape[1], len(cat)), dtype=float)
    # Iterate over each unique value
    for i, value in enumerate(cat):
        # Set the corresponding elements in the matrix to 1 where the image matches the value
        matrix[atlas == value, i] = 1

    smoothed_atlas = gaussian_filter(matrix, sigma=(smooth_value, smooth_value, 0), mode='constant')

    reduced_smoothed_atlas = np.argmax(smoothed_atlas, axis=2)

    output_atlas = np.zeros(atlas.shape)
    for i, value in enumerate(cat):
        # Set the corresponding elements in the matrix to 1 where the image matches the value
        output_atlas[reduced_smoothed_atlas == i] = value
    # print(cat, np.unique(output_atlas.flatten()))
    return output_atlas


def map_number_to_area(file_data, number):
    areas = file_data[::2]  # Extract every other line starting from the first
    index = (number - 6) // 2

    if index < 0 or index >= len(areas):
        return None  # Number is out of range
    else:
        return areas[index].strip()


def vectorize_atlas(args, atlas):
    atlas = atlas.transpose()  # because
    cat = set(np.unique(atlas.flatten()))
    cat = cat - {0, 1}
    features = []
    area_map = read_txt(args.area_id_file)
    print(cat, set(map(lambda x: map_number_to_area(area_map, int(x)), cat)))
    for value in cat:
        poly = find_contours(atlas == value, positive_orientation="high", fully_connected="high")

        polys_wm, inners_wm = build_polygon_correspondences(poly)

        # here
        s_ps = [ShapelyPolygon(x, holes=inners_wm[i]) for i, x in enumerate(polys_wm)]
        multi_poly = ShapelyMultiPolygon(s_ps)
        geom = multi_poly.__geo_interface__
        area_name = map_number_to_area(area_map, int(value))

        feat = geojson.Feature(geometry=geom, properties=dict(
            object_type="annotation",
            # TODO convert the value into the area
            classification={"name": f"area_{area_name}", "colorRGB": getIfromRGB((200, 200, 200))}
        ))
        features.append(feat)

    # Create a FeatureCollection object
    feature_collection = geojson.FeatureCollection(features)
    return feature_collection


def intersect_gm(geo):
    geo = geo.copy()
    auto_wm_multipolygon = None
    for feature in geo['features']:
        if feature['properties'].get('classification', dict()).get("name", "").startswith('auto_wm'):
            auto_wm_multipolygon = geometry.shape(feature['geometry'])
            print("found white matter")
            break

    # Perform intersection for multipolygons with classification starting with "area"
    all_features = []
    for feature in geo['features']:
        name = feature['properties'].get('classification', dict()).get("name", "")
        if name.startswith('area'):
            multipolygon = geometry.shape(feature['geometry'])
            if not multipolygon.is_valid:
                print("validity issue")
                multipolygon = make_valid(multipolygon)
            intersection = multipolygon.intersection(auto_wm_multipolygon)
            print("area", intersection.area)
            if intersection.area == 0.0:
                continue

            feat = geojson.Feature(geometry=intersection.__geo_interface__, properties=dict(
                object_type="annotation",
                # TODO convert the value into the area
                classification={"name": name, "colorRGB": getIfromRGB((200, 200, 200))}
            ))
            all_features.append(feat)
        else:
            all_features.append(feature)

    feature_collection = geojson.FeatureCollection(all_features)
    return feature_collection


def spread_polygons(geo, arr_shape, sigma=10):
    ls_arr, ls_properties = zip(*[(draw_polygon(np.zeros(arr_shape, dtype=float), shape(feat["geometry"])),
                                   feat["properties"])
                                  for feat in geo["features"]])
    ls_arr = [gaussian_filter(array, sigma=sigma) for array in ls_arr]
    arr = np.concatenate(ls_arr, axis=2)
    print(arr.shape)
    label_image = arr.argmax(axis=2) + 1
    bg_arr = arr.sum(axis=2) < 0.1
    label_image[bg_arr] = 0
    label_image = expand_labels(label_image, 1000)
    # borders
    label_image[0, :] = 0
    label_image[:, 0] = 0
    label_image[-1, :] = 0
    label_image[:, -1] = 0

    # build back the features
    features = []
    print("debug", len(ls_arr), len(ls_properties), label_image.max(), arr.shape, label_image.shape)
    for i, value in enumerate(range(1, label_image.max() + 1)):
        poly = find_contours(label_image == value, positive_orientation="high", fully_connected="high")

        polys_wm, inners_wm = build_polygon_correspondences(poly)

        # here
        s_ps = [ShapelyPolygon(x, holes=inners_wm[i]) for i, x in enumerate(polys_wm)]
        multi_poly = ShapelyMultiPolygon(s_ps)
        geom = multi_poly.__geo_interface__

        feat = geojson.Feature(geometry=geom, properties=ls_properties[i])
        features.append(feat)

    return FeatureCollection(features=features)


def run_slice(args, slice_id):
    atlas_file = build_path_histo(args.mri_atlas_dir, slice_id, "atlas_%s.png")
    atlas = read_atlas(atlas_file)

    processed_atlas = smooth_atlas(atlas)

    vectorized_atlas = vectorize_atlas(args, processed_atlas)

    output_image = str(args.mri_atlas_dir / f"annotation_on_mri_{slice_id}.png")
    image = imread(build_path_mri(args.mri_section_dir, slice_id, "raw"))
    draw_geojson_on_image(image, vectorized_atlas, output_image)

    processing_type = get_processing_type(args.schedule_steps, args.schedule_transfer_type, slice_id)
    print(processing_type)
    histo_resized_space = transform_backward_mri_histo(args, vectorized_atlas, slice_id, processing_type)

    annotation_file = build_path_histo(args.annotations_dir, slice_id, args.merged_annotations_mask)
    histo = read_histo(annotation_file)
    output_image = str(args.mri_atlas_dir / f"annotation_on_histo_{slice_id}.png")
    image = imread(build_path_histo(args.histo_dir, slice_id, args.histo_mask))

    # raster histo_space
    temp_scale = 2
    histo_resized_space = transform_rescale(1 / temp_scale, histo_resized_space)
    histo_resized_space = spread_polygons(histo_resized_space, (image.shape[0] // temp_scale,
                                                                image.shape[1] // temp_scale, 1))
    histo_resized_space = transform_rescale(temp_scale, histo_resized_space)

    histo_space = transform_backward_histo(args, histo_resized_space)

    merged_geojson = FeatureCollection(features=histo_space["features"] + histo["features"])

    merged_geojson = quickfix_multipolygon_shapely(merged_geojson)
    merged_geojson = intersect_gm(merged_geojson)
    merged_geojson_qupath = quickfix_multipolygon(merged_geojson)
    output_file = build_path_histo(args.mri_atlas_dir, slice_id, args.merged_annotations_mask)
    write_histo(merged_geojson_qupath, output_file)

    # histo_to_draw = transform_rescale(4, histo_resized_space)
    merged_geojson_shapely = quickfix_multipolygon_shapely(merged_geojson)
    histo_to_draw = transform_rescale(0.1, merged_geojson_shapely)
    draw_geojson_on_image(image, histo_to_draw, output_image)


def main(args):
    for i in range(args.start, args.end, args.step):
        try:
            run_slice(args, i)
        except FileExistsError as e:
            print(str(e))
        else:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=Path, default=Path("/media/tower/LaCie/REGISTRATION_PROJECT/"
                                                                  "TMP_PIPELINE_M148/config.ini"))
    parser.add_argument("--histo_downscale", type=float, default=None)
    parser.add_argument("--merged_annotations_mask", type=str, default=None)
    parser.add_argument("--annotations_dir", type=Path, default=None)
    parser.add_argument("--transforms_dir", type=Path, default=None)
    parser.add_argument("--mri_projections_dir", type=Path, default=None)
    parser.add_argument("--mri_section_dir", type=Path, default=None)
    parser.add_argument("--mri_atlas_dir", type=Path, default=None)
    parser.add_argument("--area_id_file", type=Path, default=None)
    parser.add_argument("--histo_dir", type=Path, default=None)
    parser.add_argument("--histo_mask", type=str, default=None)
    parser.add_argument("--schedule_steps", type=str, default=None)
    parser.add_argument("--schedule_transfer_type", type=str, default=None)
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--step", type=int, default=None)
    args_ = fill_with_config(parser)

    main(args_)
