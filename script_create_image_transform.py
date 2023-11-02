import argparse
import os
import sys
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
    extract_classification_name, get_processing_type, replace_lines_in_file, read_txt, hash_file,
    write_txt,
)
from brainseg.viz.draw import draw_polygons_from_geopandas_corrected, draw_polygons_from_geopandas_2, \
    draw_polygons_from_geopandas_3


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


def build_image_histo(histo_root, histo_annotation_root, section_id,
                      filename_mask_raw, filename_mask_annotation,
                      dict_affine_params=None,
                      predownscale_histo=0.1, redownscale_histo=0.25):
    """Create the gray images for both histology and mri"""
    if dict_affine_params is None:
        dict_affine_params = dict()

    histo_geojson = gpd.read_file(build_path_histo(
        histo_annotation_root, section_id, filename_mask_annotation))
    histo_geojson["name"] = histo_geojson["classification"].apply(extract_classification_name)

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
    image_histo = histo_concat[1].max() - histo_concat[1] + histo_concat[3] * 100 + histo_concat[4] * 100
    return image_histo


def build_image_mri(mri_root, section_id):
    mri_gm = io.imread(build_path_mri(mri_root, section_id, "gm"))
    mri_pial = io.imread(build_path_mri(mri_root, section_id, "pial"))
    mri_wm = io.imread(build_path_mri(mri_root, section_id, "wm"))
    mri_raw = io.imread(build_path_mri(mri_root, section_id, "raw"))

    mri_concat = format_mri(mri_raw, mri_pial, mri_gm, mri_wm)
    image_mri = mri_concat[1] + mri_concat[4] * 100

    return image_mri


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
    # parameter_map_affine["ResampleInterpolator"] = ["LinearInterpolator"]
    # (AutomaticScalesEstimation "true")

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
        dir_histo, dir_mri, dir_histo_annotation,
        filename_mask_raw, filename_mask_annotation, section_id,
        output_dir, hemisphere, dict_affine_params, hash_param
):
    try:
        image_histo = build_image_histo(
            dir_histo, dir_histo_annotation, section_id,
            filename_mask_raw, filename_mask_annotation,
            dict_affine_params=dict_affine_params
        )

        image_mri = build_image_mri(
            dir_mri, section_id
        )

        if image_histo.std() == 0 or image_mri.std() == 0:
            raise RuntimeError("An image has constant value, this can't be registered !")

    except Exception as e:
        # raise
        print(e)
        image_histo, image_mri = None, None

    if image_histo is None:
        return False

    section_id = str(section_id).zfill(3)

    output_all = Path(output_dir) / "manual_assistance"
    output_all.mkdir(parents=True, exist_ok=True)
    print(image_histo.shape, image_mri.shape)
    image_total = np.zeros((2000, 4000), dtype=np.uint8)
    image_total[:image_histo.shape[0], :image_histo.shape[1]] = image_histo[:, :] * 255 / image_histo.max()
    image_total[:image_mri.shape[0], 2000:2000 + image_mri.shape[1]] = image_mri[:, :] * 255 / image_mri.max()
    img = Image.fromarray(image_total)
    img.save(str(output_all / f"{section_id}.png"))

    return True


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
            dir_histo, dir_mri, dir_histo_annotation,
            filename_mask_raw, filename_mask_annotation, section_id,
            output_dir, processing_type, dict_affine_params, hash_param
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
    parser.add_argument("--manual_correction_file", type=Path, default=None)
    parser.add_argument("--histo_mask", type=str, default=None)
    parser.add_argument("--transforms_dir", type=Path, default=None)
    parser.add_argument("--schedule_steps", type=str, default=None)
    parser.add_argument("--schedule_transform_type", type=str, default=None)
    parser.add_argument("--hemisphere", type=str, default=None)
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--step", type=int, default=None)

    args_ = fill_with_config(parser)
    main(args_)
