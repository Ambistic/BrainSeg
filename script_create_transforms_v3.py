import argparse
import os
import warnings
from pathlib import Path

import itk
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from shapely.ops import transform
from skimage import io, measure
from skimage.transform import rescale
import geopandas as gpd
from tqdm import tqdm
import re

from brainseg.config import fill_with_config
from brainseg.geo import create_affine_mapping
from brainseg.math import compute_angle
from brainseg.parser import parse_dict_param
from brainseg.path import build_path_mri, build_path_histo
from brainseg.utils import extract_classification_name, get_processing_type
from brainseg.viz.draw import draw_polygons_from_geopandas, draw_polygon, draw_polygons


def binarize_mask(mask):
    mask = mask > max(mask.max() / 2, 1 / 2)
    return mask.astype(int)


def expand_mask_dims(mask):
    return np.expand_dims(mask, axis=2)


def binarize_white_mask(mask):
    mask = mask[:, :, 0]
    # mask = mask.max() - mask  # reverse mask
    mask = mask > max(mask.max() / 2, 1 / 2)
    return mask.astype(int)


def replace_lines_in_file(file_path, pattern, replacement):
    # Read the file
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Replace lines matching the pattern
    new_lines = [re.sub(pattern, replacement, line) for line in lines]

    # Write the modified lines back to the file
    with open(file_path, 'w') as file:
        file.writelines(new_lines)


def format_histo_mri(histo_raw, histo_pial, histo_gm,
                     mri_raw, mri_pial, mri_gm, mri_wm,
                     histo_rescale_factor=1.):
    # binarize
    histo_pial = expand_mask_dims(binarize_mask(histo_pial))
    histo_gm = expand_mask_dims(binarize_mask(histo_gm))

    mri_pial = expand_mask_dims(binarize_white_mask(mri_pial))
    mri_gm = expand_mask_dims(binarize_white_mask(mri_gm))
    mri_wm = expand_mask_dims(binarize_white_mask(mri_wm))

    # maskify
    histo_raw = histo_raw * histo_pial + 255 * (1 - histo_pial)
    mri_raw = mri_raw * mri_pial

    # rescale
    histo_raw = rescale(histo_raw.astype(float), histo_rescale_factor, anti_aliasing=False, channel_axis=2)
    histo_pial = rescale(histo_pial.astype(float), histo_rescale_factor, anti_aliasing=False)
    histo_gm = rescale(histo_gm.astype(float), histo_rescale_factor, anti_aliasing=False)

    # format
    histo_concat = [img[:, :, i] for img in [histo_raw, histo_pial, histo_gm] for i in range(img.shape[2])]
    mri_concat = [img[:, :, i] for img in [mri_raw, mri_pial, mri_gm, mri_wm] for i in range(img.shape[2])]

    return histo_concat, mri_concat


def build_images(histo_root, histo_annotation_root, mri_root, section_id,
                 filename_mask_raw, filename_mask_annotation,
                 predownscale_histo=0.1, redownscale_histo=0.25):
    """Create the gray images for both histology and mri"""
    mri_gm = io.imread(build_path_mri(mri_root, section_id, "gm"))
    mri_pial = io.imread(build_path_mri(mri_root, section_id, "pial"))
    mri_wm = io.imread(build_path_mri(mri_root, section_id, "wm"))
    mri_raw = io.imread(build_path_mri(mri_root, section_id, "raw"))

    histo_geojson = gpd.read_file(build_path_histo(
        histo_annotation_root, section_id, filename_mask_annotation))
    histo_geojson["name"] = histo_geojson["classification"].apply(extract_classification_name)
    histo_raw = io.imread(build_path_histo(histo_root, section_id, filename_mask_raw))

    histo_raw = rescale(histo_raw.astype(float), redownscale_histo, anti_aliasing=False, channel_axis=2)

    arr = np.zeros((histo_raw.shape[0], histo_raw.shape[1]))

    histo_pial = draw_polygons_from_geopandas(arr, histo_geojson[histo_geojson["name"] == "auto_outline"],
                                              predownscale_histo * redownscale_histo, reverse=True)
    histo_gm = draw_polygons_from_geopandas(arr, histo_geojson[histo_geojson["name"] == "auto_wm"],
                                            predownscale_histo * redownscale_histo, reverse=True)

    histo_concat, mri_concat = format_histo_mri(histo_raw, histo_pial, histo_gm,
                                                mri_raw, mri_pial, mri_gm, mri_wm)

    image_histo = histo_concat[1].max() - histo_concat[1] + histo_concat[3] * 100 + histo_concat[4] * 100
    image_mri = mri_concat[1] + mri_concat[4] * 100

    return image_histo, image_mri


def compute_transform(image_histo, image_mri, output_directory, hemisphere):
    fixed = itk.GetImageFromArray(image_histo.astype(np.float32), is_vector=False)
    moving = itk.GetImageFromArray(image_mri.astype(np.float32), is_vector=False)

    parameter_object = itk.ParameterObject.New()
    parameter_map_affine = parameter_object.GetDefaultParameterMap('affine')
    parameter_map_bspline = parameter_object.GetDefaultParameterMap('bspline')

    parameter_map_affine["MaximumNumberOfIterations"] = ['1024']
    parameter_map_affine["NumberOfResolutions"] = ['6.000000']
    parameter_map_affine["Metric"] = ["AdvancedNormalizedCorrelation"]
    parameter_map_affine["NumberOfSpatialSamples"] = ["50000"]

    parameter_map_affine["AutomaticTransformInitialization"] = ["true"]
    parameter_map_affine["AutomaticTransformInitializationMethod"] = ["CenterOfGravity"]

    parameter_map_bspline["MaximumNumberOfIterations"] = ['1024']
    parameter_map_bspline["NumberOfResolutions"] = ['4.000000']
    parameter_map_bspline["Metric"] = ["AdvancedMattesMutualInformation", "TransformBendingEnergyPenalty"]

    parameter_map_bspline["Metric0Weight"] = ["1"]
    parameter_map_bspline["Metric1Weight"] = ["500"]
    parameter_map_bspline["NumberOfSpatialSamples"] = ["50000"]

    parameter_object.AddParameterMap(parameter_map_affine)
    parameter_object.AddParameterMap(parameter_map_bspline)

    # Call registration function and specify output directory
    result_image, result_transform_parameters = itk.elastix_registration_method(
        fixed, moving,
        parameter_object=parameter_object,
        output_directory=str(output_directory),
        # number_of_threads=16,
    )

    if hemisphere == "right" or hemisphere == "both":
        compute_side_transform(image_mri, image_histo, parameter_map_affine, parameter_map_bspline,
                               result_transform_parameters, output_directory, side="right")
    if hemisphere == "left" or hemisphere == "both":
        compute_side_transform(image_mri, image_histo, parameter_map_affine, parameter_map_bspline,
                               result_transform_parameters, output_directory, side="left")

    return result_image, result_transform_parameters


def compute_side_transform(image_mri, image_histo, parameter_map_affine, parameter_map_bspline,
                           result_transform_parameters, output_directory, side="right"):
    image_mri_side = image_mri.copy()
    if side == "right":
        image_mri_side[:, :image_mri_side.shape[0] // 2] = 0
    elif side == "left":
        image_mri_side[:, image_mri_side.shape[0] // 2:] = 0
    else:
        raise ValueError("side should be either right or left")

    moving_side = itk.GetImageFromArray(image_mri_side.astype(np.float32), is_vector=False)

    side_histo = itk.transformix_filter(
        moving_side,
        result_transform_parameters
    )

    side_histo = np.array(side_histo)
    mask_side_histo = side_histo > 20  # the threshold for foreground is 100, but we prefer to be a bit permissive

    # mask_right_histo = binary_dilation(mask_right_histo, iterations=3).astype(bool)
    image_histo_side = image_histo.copy()
    image_histo_side[~mask_side_histo] = 0

    fixed_side = itk.GetImageFromArray(image_histo_side.astype(np.float32), is_vector=False)

    parameter_map_affine["TransformParameters"] = \
        result_transform_parameters.GetParameterMap(0)["TransformParameters"]
    parameter_map_bspline["TransformParameters"] = \
        result_transform_parameters.GetParameterMap(1)["TransformParameters"]

    parameter_object = itk.ParameterObject.New()
    parameter_object.AddParameterMap(parameter_map_affine)
    parameter_object.AddParameterMap(parameter_map_bspline)

    result_image_side, result_transform_parameters_side = itk.elastix_registration_method(
        fixed_side, moving_side,
        parameter_object=parameter_object,
        output_directory=os.path.join(str(output_directory), side),
    )

    format_to_grey_image(result_image_side).save(output_directory / side / "image.png")


def compute_inverse_transform(image_histo, input_directory, output_directory, hemisphere):
    fixed = itk.GetImageFromArray(image_histo.astype(np.float32), is_vector=False)

    # Import Default Parameter Map and adjust parameters
    parameter_object = itk.ParameterObject.New()
    parameter_map_affine = parameter_object.GetDefaultParameterMap('affine')
    parameter_map_bspline = parameter_object.GetDefaultParameterMap('bspline')
    parameter_map_bspline['HowToCombineTransforms'] = ['Compose']
    parameter_map_affine['HowToCombineTransforms'] = ['Compose']
    parameter_map_bspline['Metric'] = ['DisplacementMagnitudePenalty']
    parameter_map_affine['Metric'] = ['DisplacementMagnitudePenalty']
    parameter_object.AddParameterMap(parameter_map_affine)
    parameter_object.AddParameterMap(parameter_map_bspline)

    # Call registration function with transform parameters of normal misc run as initial transform
    # on fixed image to fixed image registration. In ITKElastix there is not option to pass the
    # result_transform_parameters
    # as a python object yet, the initial transform can only be passed as a .txt file to
    # initial_transform_parameter_file_name.
    # Elastix also writes the transform parameter file to a .txt file if an output directory is specified.
    # Make sure to give the correct TransformParameterFile.txt to misc if multiple parameter maps are used.
    pattern = r'\(InitialTransformParametersFileName ".*"\)'
    replace = r'(InitialTransformParametersFileName "NoInitialTransform")'

    inverse_image, inverse_transform_parameters = itk.elastix_registration_method(
        fixed, fixed,
        parameter_object=parameter_object,
        initial_transform_parameter_file_name=str(input_directory / 'TransformParameters.1.txt'),
        output_directory=str(output_directory),
    )
    replace_lines_in_file(str(output_directory / 'TransformParameters.0.txt'), pattern, replace)

    if hemisphere == "right" or hemisphere == "both":
        inverse_image_right, inverse_transform_parameters_right = itk.elastix_registration_method(
            fixed, fixed,
            parameter_object=parameter_object,
            initial_transform_parameter_file_name=str(input_directory / "right" / 'TransformParameters.1.txt'),
            output_directory=str(output_directory / "right"),
        )
        replace_lines_in_file(str(output_directory / "right" / 'TransformParameters.0.txt'), pattern, replace)

    if hemisphere == "left" or hemisphere == "both":
        inverse_image_left, inverse_transform_parameters_left = itk.elastix_registration_method(
            fixed, fixed,
            parameter_object=parameter_object,
            initial_transform_parameter_file_name=str(input_directory / "left" / 'TransformParameters.1.txt'),
            output_directory=str(output_directory / "left"),
        )
        replace_lines_in_file(str(output_directory / "left" / 'TransformParameters.0.txt'), pattern, replace)

    # Adjust inverse transform parameters object
    inverse_transform_parameters.SetParameter(
        0, "InitialTransformParametersFileName", "NoInitialTransform")  # TODO save this one

    return inverse_image, inverse_transform_parameters


# unused
def apply_transform(image, transform_path, output_dir):
    fixed = itk.GetImageFromArray(image.astype(np.float32), is_vector=False)
    # Load Transformix Object
    transformix_object = itk.TransformixFilter.New(fixed)
    transform_parameters = itk.ParameterObject.ReadParameterFile(transform_path)
    transformix_object.SetTransformParameterObject(transform_parameters)

    # Set advanced options
    transformix_object.SetComputeDeformationField(True)

    # Set output directory for spatial jacobian and its determinant,
    # default directory is current directory.
    transformix_object.SetOutputDirectory(output_dir)

    # Update object (required)
    transformix_object.UpdateLargestPossibleRegion()

    # Results of Transformation
    result_image_transformix = transformix_object.GetOutput()
    deformation_field = transformix_object.GetOutputDeformationField()

    return result_image_transformix, deformation_field


def format_to_grey_image(image_array):
    image_array = np.array(image_array)
    # print(image_array.dtype, np.max(image_array))
    image_array = image_array / np.max(image_array) * 255.
    return Image.fromarray(image_array.astype(int).astype(np.uint8()))


def get_connected_components(image):
    # Label connected components
    labeled_image = measure.label(image, connectivity=2)

    # Get the number of connected components
    num_components = np.max(labeled_image)

    # Get the properties of each connected component
    components = []
    for component_label in range(1, num_components + 1):
        component = labeled_image == component_label
        components.append(component)

    return components


def make_polygon_ordering(polygons_outline):
    centroids = np.array([polygon.centroid.coords[0] for polygon in polygons_outline])
    centroids -= centroids.mean(axis=0)
    angles = [compute_angle(x, y) for x, y in centroids]

    return np.argsort(angles)


def validate_no_overflow(polygons, buf=50, raises=True):
    # Check for intersections between polygons
    for i, polygon1 in enumerate(polygons):
        for j, polygon2 in enumerate(polygons[i+1:], start=i+1):
            if polygon1.buffer(buf).intersects(polygon2.buffer(buf)):
                if raises:
                    raise ValueError(f"Polygons {i} and {j} intersect")
                else:
                    warnings.warn(f"Polygons {i} and {j} intersect")


def create_transforms(
        dir_histo, dir_mri, dir_histo_annotation,
        filename_mask_raw, filename_mask_annotation, section_id,
        output_dir, hemisphere
):
    mri_pial = io.imread(build_path_mri(dir_mri, section_id, "pial"))
    # build polygons
    histo_geojson = gpd.read_file(build_path_histo(
        dir_histo_annotation, section_id, filename_mask_annotation))
    histo_geojson["name"] = histo_geojson["classification"].apply(extract_classification_name)

    outlines = histo_geojson[histo_geojson["name"] == "auto_outline"].iloc[0].explode()

    polygons_outline = list(outlines["geometry"])

    # identify polygons
    order = make_polygon_ordering(polygons_outline)
    ordered_polygons = [polygons_outline[index] for index in order]

    dict_affine_params = parse_dict_param("352_1_N_15_-1000_-500, 352_0_N_-30_3500_-1000")

    print(dict_affine_params)

    transformed_polygons = []

    for i, poly in enumerate(ordered_polygons):
        if dict_affine_params.get((section_id, i)) is not None:
            flip, rotation_angle, shift_x, shift_y = dict_affine_params.get((section_id, i))
            center_x, center_y = poly.centroid.coords[0]
            map_func = create_affine_mapping(flip=flip, rot=rotation_angle,
                                             shift_x=shift_x, shift_y=shift_y,
                                             center_x=center_x, center_y=center_y)
            poly = transform(map_func, poly)

        transformed_polygons.append(poly)

    validate_no_overflow(transformed_polygons, raises=False)

    arr = np.zeros((1000, 1000))
    arr_ori = draw_polygons(arr, ordered_polygons, rescale=0.02)
    arr_trs = draw_polygons(arr, transformed_polygons, rescale=0.02)
    plt.subplot(1, 3, 1)
    plt.imshow(arr_ori.T)
    plt.subplot(1, 3, 2)
    plt.imshow(arr_trs.T)
    plt.subplot(1, 3, 3)
    plt.imshow(mri_pial)
    plt.show()

    # import pdb; pdb.set_trace()

    return

    try:
        image_histo, image_mri = build_images(
            dir_histo, dir_histo_annotation, dir_mri, section_id,
            filename_mask_raw, filename_mask_annotation
        )
    except Exception as e:
        print(e)
        image_histo, image_mri = None, None

    if image_histo is None:
        return False

    section_id = str(section_id).zfill(3)
    output_folder_forward = Path(output_dir) / (section_id + "_forward")
    output_folder_backward = Path(output_dir) / (section_id + "_backward")

    output_folder_forward.mkdir(parents=True, exist_ok=True)
    output_folder_backward.mkdir(parents=True, exist_ok=True)

    (output_folder_forward / "right").mkdir(parents=True, exist_ok=True)
    (output_folder_backward / "right").mkdir(parents=True, exist_ok=True)

    (output_folder_forward / "left").mkdir(parents=True, exist_ok=True)
    (output_folder_backward / "left").mkdir(parents=True, exist_ok=True)

    # create transform
    image_forward, _ = compute_transform(image_histo, image_mri, output_folder_forward, hemisphere)
    image_backward, _ = compute_inverse_transform(image_histo, output_folder_forward, output_folder_backward,
                                                  hemisphere)

    # export quality controls
    format_to_grey_image(image_histo).save(output_folder_forward / "image_histo.png")
    format_to_grey_image(image_mri).save(output_folder_forward / "image_mri.png")
    format_to_grey_image(image_forward).save(output_folder_forward / "image_forward.png")
    format_to_grey_image(image_backward).save(output_folder_forward / "image_backward.png")

    return True


def create_transform_from_dirs(args, dir_histo, dir_mri, dir_histo_annotation,
                               filename_mask_raw, filename_mask_annotation,
                               output_dir, start=0, end=500, step=1):
    sections_made = []
    for section_id in tqdm(range(start, end, step)):
        processing_type = get_processing_type(args.schedule_steps, args.schedule_transform_type, section_id)

        is_made = create_transforms(
            dir_histo, dir_mri, dir_histo_annotation,
            filename_mask_raw, filename_mask_annotation, section_id,
            output_dir, processing_type
        )
        if is_made:
            sections_made.append(section_id)
    return sections_made


def main(args):
    sections = create_transform_from_dirs(args, args.histo_dir, args.mri_section_dir, args.annotations_dir,
                                          args.histo_mask, args.full_annotations_mask,
                                          args.transforms_dir,
                                          start=args.start, end=args.end, step=args.step)

    print(sections)
    print(f'{len(sections)} made')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=Path, default=Path("/media/tower/LaCie/REGISTRATION_PROJECT/"
                                                                  "TMP_PIPELINE_M148/config.ini"))
    parser.add_argument("--histo_dir", type=Path, default=None)
    parser.add_argument("--mri_section_dir", type=Path, default=None)
    parser.add_argument("--annotations_dir", type=Path, default=None)
    parser.add_argument("--full_annotations_mask", type=str, default=None)
    parser.add_argument("--histo_mask", type=str, default=None)
    parser.add_argument("--transforms_dir", type=Path, default=None)
    parser.add_argument("--schedule_steps", type=str, default=None)
    parser.add_argument("--schedule_transform_type", type=str, default=None)
    parser.add_argument("--hemisphere", type=str, default=None)
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--step", type=int, default=None)

    args_ = fill_with_config(parser)
    args_.start = 352
    args_.end = 353
    main(args_)
