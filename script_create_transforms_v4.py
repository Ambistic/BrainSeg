import argparse
import json
import os
import traceback
from collections import defaultdict
from pathlib import Path
import itk
from PIL import Image
import numpy as np
from skimage import io
from skimage.transform import rescale
import geopandas as gpd
from tqdm import tqdm
import matplotlib.pyplot as plt

from brainseg.config import fill_with_config
from brainseg.geo import polygons_to_geopandas
from brainseg.misc.image_geometry import image_manual_correction
from brainseg.misc.manual_correction import process_pial_gm_manual_correction
from brainseg.parser import parse_dict_param
from brainseg.path import build_path_mri, build_path_histo
from brainseg.utils import (
    get_processing_type, replace_lines_in_file, read_txt, hash_file,
    write_txt, calculate_name,
)
from brainseg.viz.draw import draw_polygons_from_geopandas_3


LOG_ITK_TO_CONSOLE = False


def process_histo_binary(histo):
    image_histo = histo[3] * 100 + histo[4] * 100
    return image_histo


def process_histo_original(histo):
    image_histo = histo[1].max() - histo[1] + histo[3] * 100 + histo[4] * 100
    return image_histo


def process_histo_contrasted(histo):
    image_histo = (histo[0] * 0.4 + histo[1] * 0.4 + histo[2] * 0.2) * 1 + histo[3] * 180 - histo[4] * 50 - 250
    image_histo = np.clip(image_histo, 0, 255)
    return image_histo


def process_histo_raw(histo):
    image_histo = histo[0] * 0.4 + histo[1] * 0.4 + histo[2] * 0.2
    return image_histo


def process_mri_binary(mri):
    image_mri = mri[4] * 100
    return image_mri


def process_mri_original(mri):
    image_mri = mri[1] + mri[4] * 100
    return image_mri


def process_mri_contrasted(mri):
    image_mri = mri[1] - mri[4] * 20
    image_mri = np.clip(image_mri, 0, 255)
    return image_mri


def process_mri_raw(mri):
    image_mri = mri[1]
    return image_mri


MAP_IMAGE_TYPE = dict(
    binary=(process_histo_binary, process_mri_binary),
    original=(process_histo_original, process_mri_original),
    contrasted=(process_histo_contrasted, process_mri_contrasted),
    raw=(process_histo_raw, process_mri_raw),
)


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


def format_histo(histo_raw, histo_pial, histo_gm,
                 histo_rescale_factor=1.):
    # binarize
    histo_pial = expand_mask_dims(binarize_mask(histo_pial))
    histo_gm = expand_mask_dims(binarize_mask(histo_gm))

    # maskify
    histo_raw = histo_raw * histo_pial + 255 * (1 - histo_pial)

    # rescale
    histo_raw = rescale(histo_raw.astype(float), histo_rescale_factor, anti_aliasing=False, channel_axis=2)
    histo_pial = rescale(histo_pial.astype(float), histo_rescale_factor, anti_aliasing=False)
    histo_gm = rescale(histo_gm.astype(float), histo_rescale_factor, anti_aliasing=False)

    # format
    histo_concat = [img[:, :, i] for img in [histo_raw, histo_pial, histo_gm] for i in range(img.shape[2])]

    return histo_concat


def format_mri(mri_raw, mri_pial, mri_gm, mri_wm,):
    mri_pial = expand_mask_dims(binarize_white_mask(mri_pial))
    mri_gm = expand_mask_dims(binarize_white_mask(mri_gm))
    mri_wm = expand_mask_dims(binarize_white_mask(mri_wm))
    mri_raw = mri_raw * mri_pial
    mri_concat = [img[:, :, i] for img in [mri_raw, mri_pial, mri_gm, mri_wm] for i in range(img.shape[2])]

    return mri_concat


def build_image_histo(args, histo_root, histo_annotation_root, section_id,
                      filename_mask_raw, filename_mask_annotation,
                      dict_affine_params=None,
                      predownscale_histo=0.1, redownscale_histo=0.25):
    """Create the gray images for both histology and mri"""
    if dict_affine_params is None:
        dict_affine_params = dict()

    histo_geojson = gpd.read_file(build_path_histo(
        histo_annotation_root, section_id, filename_mask_annotation))
    # depending on the version, the name is already there
    calculate_name(histo_geojson)

    ordered_pial, _, params, _, transformed_pial, transformed_gm = process_pial_gm_manual_correction(
        dict_affine_params, histo_geojson, section_id)

    histo_raw = io.imread(build_path_histo(histo_root, section_id, filename_mask_raw))
    histo_raw = rescale(histo_raw.astype(float), redownscale_histo, anti_aliasing=False, channel_axis=2)

    margin = (200, 100)
    shape = (histo_raw.shape[0] + margin[1], histo_raw.shape[1] + margin[0])

    arr = np.zeros(shape)

    histo_pial = draw_polygons_from_geopandas_3(arr.copy(), polygons_to_geopandas(transformed_pial),
                                                predownscale_histo * redownscale_histo, reverse=True)
    histo_gm = draw_polygons_from_geopandas_3(arr.copy(), polygons_to_geopandas(transformed_gm),
                                              predownscale_histo * redownscale_histo, reverse=True)

    histo_mod = image_manual_correction(histo_raw, params, ordered_pial, margin=margin,
                                        swap_xy=True, background=0, scale=predownscale_histo * redownscale_histo)

    histo_concat = format_histo(histo_mod, histo_pial, histo_gm)
    return MAP_IMAGE_TYPE[args.format_registration_type][0](histo_concat)


def build_image_mri(args, mri_root, section_id):
    mri_gm = io.imread(build_path_mri(mri_root, section_id, "gm"))
    mri_pial = io.imread(build_path_mri(mri_root, section_id, "pial"))
    mri_wm = io.imread(build_path_mri(mri_root, section_id, "wm"))
    mri_raw = io.imread(build_path_mri(mri_root, section_id, "raw"))

    mri_concat = format_mri(mri_raw, mri_pial, mri_gm, mri_wm)
    return MAP_IMAGE_TYPE[args.format_registration_type][1](mri_concat)


def write_point_txt(point_file, points):
    with open(point_file, "w") as file:
        print("point", file=file)
        print(len(points), file=file)
        for p in points:
            print(" ".join(map(str, p)), file=file)


def build_point_set_args(assistance_file):
    with open(assistance_file, 'r') as file:
        data = json.load(file)

    # Initialize a dictionary to store label names and their associated coordinates
    label_coordinates = defaultdict(list)

    # Iterate through the "shapes" array
    for shape in data['shapes']:
        label = shape['label']
        coordinates = shape['points'][0]  # Assuming there is only one point per shape
        label_coordinates[label].append(coordinates)

    # Print or process the extracted label names and coordinates
    point_histo, point_mri = [], []
    for label, coordinates in label_coordinates.items():
        # check
        if len(coordinates) != 2:
            print(f"Bad number of point for label {label} : {len(coordinates)}")
            continue

        point1, point2 = coordinates
        if len(point1) != 2 or len(point2) != 2:
            print("Bad coordinates dimension")
            continue

        if point1[0] >= 2000:
            point1, point2 = point2, point1
        elif point2[0] < 2000:
            print(f"Two points on the same side for label {label}")
            continue

        if point1[0] >= 2000:
            print(f"Two points on the same side for label {label}")
            continue

        point_histo.append(point1)
        point_mri.append([point2[0] - 2000, point2[1]])
        print(f"Label: {label}, Coordinates: {coordinates}")

    # export to files and construct the dict
    histo_point_file = assistance_file.parent / f"{assistance_file.stem}_histo.txt"
    mri_point_file = assistance_file.parent / f"{assistance_file.stem}_mri.txt"
    write_point_txt(histo_point_file, point_histo)
    write_point_txt(mri_point_file, point_mri)
    return dict(fixed_point_set_file_name=str(histo_point_file),
                moving_point_set_file_name=str(mri_point_file))


def compute_transform(image_histo, image_mri, output_directory, hemisphere, section_id, assistance_dir):
    # test if manual assistance
    assistance_file = assistance_dir / f"{section_id}.json"
    manual_assistance = assistance_file.exists()
    if manual_assistance:
        point_set_args = build_point_set_args(assistance_file)
    else:
        point_set_args = dict()
    # raise
    fixed = itk.GetImageFromArray(image_histo.astype(np.float32), is_vector=False)
    moving = itk.GetImageFromArray(image_mri.astype(np.float32), is_vector=False)

    parameter_object = itk.ParameterObject.New()
    parameter_map_affine = parameter_object.GetDefaultParameterMap('affine')
    parameter_map_bspline = parameter_object.GetDefaultParameterMap('bspline')

    parameter_map_affine["MaximumNumberOfIterations"] = ['1024']
    parameter_map_affine["NumberOfResolutions"] = ['6.000000']
    if manual_assistance and False:  # keep as False or breaks the optimization
        parameter_map_affine['Registration'] = [
            'MultiMetricMultiResolutionRegistration']
        parameter_map_affine["Metric"] = ["AdvancedNormalizedCorrelation",
                                          "CorrespondingPointsEuclideanDistanceMetric"]
    else:
        parameter_map_affine["Metric"] = ["AdvancedNormalizedCorrelation"]

    parameter_map_affine["NumberOfSpatialSamples"] = ["50000"]

    # parameter_map_affine["ResampleInterpolator"] = ["LinearInterpolator"]
    # (AutomaticScalesEstimation "true")

    parameter_map_affine["AutomaticTransformInitialization"] = ["true"]
    parameter_map_affine["AutomaticTransformInitializationMethod"] = ["CenterOfGravity"]

    parameter_map_bspline["MaximumNumberOfIterations"] = ['1024']
    parameter_map_bspline["NumberOfResolutions"] = ['4.000000']
    if manual_assistance:
        parameter_map_bspline['Registration'] = [
            'MultiMetricMultiResolutionRegistration']
        parameter_map_bspline["Metric"] = ["AdvancedMattesMutualInformation", "TransformBendingEnergyPenalty",
                                           "CorrespondingPointsEuclideanDistanceMetric"]
        parameter_map_bspline["Metric2Weight"] = ["0.01"]
    else:
        parameter_map_bspline["Metric"] = ["AdvancedMattesMutualInformation", "TransformBendingEnergyPenalty"]

    parameter_map_bspline["Metric0Weight"] = ["1"]
    parameter_map_bspline["Metric1Weight"] = ["500"]
    parameter_map_bspline["NumberOfSpatialSamples"] = ["50000"]

    parameter_object.AddParameterMap(parameter_map_affine)
    parameter_object.AddParameterMap(parameter_map_bspline)

    # Call registration function and specify output directory
    print(point_set_args)
    result_images = dict()
    result_image, result_transform_parameters = itk.elastix_registration_method(
        fixed, moving,
        parameter_object=parameter_object,
        output_directory=str(output_directory),
        log_to_console=LOG_ITK_TO_CONSOLE,
        **point_set_args,
        # number_of_threads=16,
    )
    result_images["both"] = result_image

    if hemisphere == "right" or hemisphere == "both":
        image_right, _ = compute_side_transform(image_mri, image_histo, parameter_map_affine, parameter_map_bspline,
                                                result_transform_parameters, output_directory, side="right",
                                                point_set_args=point_set_args)
        result_images["right"] = image_right
    if hemisphere == "left" or hemisphere == "both":
        image_left, _ = compute_side_transform(image_mri, image_histo, parameter_map_affine, parameter_map_bspline,
                                               result_transform_parameters, output_directory, side="left",
                                               point_set_args=point_set_args)
        result_images["left"] = image_left

    return result_images, result_transform_parameters


def compute_side_transform(image_mri, image_histo, parameter_map_affine, parameter_map_bspline,
                           result_transform_parameters, output_directory, side="right", point_set_args=None):
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

    if point_set_args is None:
        point_set_args = dict()

    result_image_side, result_transform_parameters_side = itk.elastix_registration_method(
        fixed_side, moving_side,
        parameter_object=parameter_object,
        output_directory=os.path.join(str(output_directory), side),
        log_to_console=LOG_ITK_TO_CONSOLE,
        **point_set_args
    )

    format_to_grey_image(result_image_side).save(output_directory / side / "image.png")
    return result_image_side, result_transform_parameters_side


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

    result_images = dict()

    inverse_image, inverse_transform_parameters = itk.elastix_registration_method(
        fixed, fixed,
        parameter_object=parameter_object,
        initial_transform_parameter_file_name=str(input_directory / 'TransformParameters.1.txt'),
        output_directory=str(output_directory),
        log_to_console=LOG_ITK_TO_CONSOLE,
    )
    replace_lines_in_file(str(output_directory / 'TransformParameters.0.txt'), pattern, replace)
    result_images["both"] = inverse_image

    if hemisphere == "right" or hemisphere == "both":
        inverse_image_right, inverse_transform_parameters_right = itk.elastix_registration_method(
            fixed, fixed,
            parameter_object=parameter_object,
            initial_transform_parameter_file_name=str(input_directory / "right" / 'TransformParameters.1.txt'),
            output_directory=str(output_directory / "right"),
            log_to_console=LOG_ITK_TO_CONSOLE,
        )
        replace_lines_in_file(str(output_directory / "right" / 'TransformParameters.0.txt'), pattern, replace)
        result_images["right"] = inverse_image_right

    if hemisphere == "left" or hemisphere == "both":
        inverse_image_left, inverse_transform_parameters_left = itk.elastix_registration_method(
            fixed, fixed,
            parameter_object=parameter_object,
            initial_transform_parameter_file_name=str(input_directory / "left" / 'TransformParameters.1.txt'),
            output_directory=str(output_directory / "left"),
            log_to_console=LOG_ITK_TO_CONSOLE,
        )
        replace_lines_in_file(str(output_directory / "left" / 'TransformParameters.0.txt'), pattern, replace)
        result_images["left"] = inverse_image_left

        # Adjust inverse transform parameters object
    inverse_transform_parameters.SetParameter(
        0, "InitialTransformParametersFileName", "NoInitialTransform")

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


def create_transforms(
        args, dir_histo, dir_mri, dir_histo_annotation,
        filename_mask_raw, filename_mask_annotation, section_id,
        output_dir, hemisphere, dict_affine_params, hash_param
):
    try:
        image_histo = build_image_histo(
            args, dir_histo, dir_histo_annotation, section_id,
            filename_mask_raw, filename_mask_annotation,
            dict_affine_params=dict_affine_params
        )

        image_mri = build_image_mri(
            args, dir_mri, section_id
        )

        if image_histo.std() == 0 or image_mri.std() == 0:
            raise RuntimeError(f"An image ({section_id}) has constant value, this can't be registered !")

        # check if not too few pixels are not background
        if (image_mri != image_mri.min()).mean() < 0.01:
            raise RuntimeError(f"An image ({section_id}) has almost constant value, this can't be registered !")

    except Exception as e:
        # raise
        print(e)
        traceback.print_exc()
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

    if False:
        mri_raw = io.imread(build_path_mri(dir_mri, section_id, "raw"))
        output_all = Path(output_dir) / "review"
        output_all.mkdir(parents=True, exist_ok=True)

        export_review_images(image_histo, image_histo, image_mri, mri_raw, output_all, section_id, "both")
        return

    # create transform
    images_forward, _ = compute_transform(image_histo, image_mri, output_folder_forward, hemisphere, section_id,
                                         Path(output_dir) / "manual_assistance")
    image_backward, _ = compute_inverse_transform(image_histo, output_folder_forward, output_folder_backward,
                                                  hemisphere)

    # export quality controls
    format_to_grey_image(image_histo).save(output_folder_forward / "image_histo.png")
    format_to_grey_image(image_mri).save(output_folder_forward / "image_mri.png")
    format_to_grey_image(images_forward["both"]).save(output_folder_forward / "image_forward.png")
    format_to_grey_image(image_backward).save(output_folder_forward / "image_backward.png")

    mri_raw = io.imread(build_path_mri(dir_mri, section_id, "raw"))
    output_all = Path(output_dir) / "review"
    output_all.mkdir(parents=True, exist_ok=True)

    export_review_images(images_forward["both"], image_histo, image_mri, mri_raw, output_all, section_id, "both")
    if "right" in images_forward:
        export_review_images(images_forward["right"], image_histo, image_mri, mri_raw, output_all, section_id, "right")
    if "left" in images_forward:
        export_review_images(images_forward["left"], image_histo, image_mri, mri_raw, output_all, section_id, "left")

    write_txt(output_folder_forward / "hash.txt", hash_param)

    return True


def export_review_images(image_forward, image_histo, image_mri, mri_raw, output_all, section_id, name="both"):
    plt.figure(figsize=(12, 12))
    plt.subplot(2, 2, 1)
    plt.imshow(image_histo)
    plt.title(f"{np.asarray(image_histo).min()}:{np.asarray(image_histo).max()}")
    plt.colorbar()
    plt.subplot(2, 2, 2)
    plt.imshow(image_mri)
    plt.title(f"{np.asarray(image_mri).min()}:{np.asarray(image_mri).max()}")
    plt.colorbar()
    plt.subplot(2, 2, 3)
    plt.imshow(image_forward)
    plt.title(f"{np.asarray(image_forward).min()}:{np.asarray(image_forward).max()}")
    plt.colorbar()
    plt.subplot(2, 2, 4)
    plt.imshow(mri_raw)
    plt.savefig(str(output_all / f"{section_id}_{name}.png"))


def create_transform_from_dirs(args, dir_histo, dir_mri, dir_histo_annotation,
                               filename_mask_raw, filename_mask_annotation,
                               output_dir, start=0, end=500, step=1):
    sections_made = []
    param_data = read_txt(args.manual_correction_file)
    hash_param = hash_file(args.manual_correction_file)
    dict_affine_params = parse_dict_param(",".join(param_data))
    for section_id in tqdm(range(start, end, step)):
        processing_type = get_processing_type(args.schedule_steps, args.schedule_transform_type, section_id)

        is_made = create_transforms(
            args, dir_histo, dir_mri, dir_histo_annotation,
            filename_mask_raw, filename_mask_annotation, section_id,
            output_dir, processing_type, dict_affine_params, hash_param
        )
        if is_made:
            sections_made.append(section_id)
    return sections_made


def main(args):
    sections = create_transform_from_dirs(args, args.histo_dir, args.mri_sections_dir, args.annotations_dir,
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
    parser.add_argument("--mri_sections_dir", type=Path, default=None)
    parser.add_argument("--annotations_dir", type=Path, default=None)
    parser.add_argument("--full_annotations_mask", type=str, default=None)
    parser.add_argument("--manual_correction_file", type=Path, default=None)
    parser.add_argument("--histo_mask", type=str, default=None)
    parser.add_argument("--transforms_dir", type=Path, default=None)
    parser.add_argument("--schedule_steps", type=str, default=None)
    parser.add_argument("--schedule_transform_type", type=str, default=None)
    parser.add_argument("--hemisphere", type=str, default=None)
    parser.add_argument("--format_registration_type", type=str, default=None)
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--step", type=int, default=None)

    args_ = fill_with_config(parser)
    main(args_)
