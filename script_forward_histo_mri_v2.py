import argparse
import traceback
import warnings
from pathlib import Path

import numpy as np
from skimage.io import imread
import geopandas as gpd

from brainseg.config import fill_with_config
from brainseg.geo import record_point_coordinates, transform_from_dict, transform_forward_histo, flatten_geojson, \
    transform_from_manual_correction, transform_rescale, explode_multipolygon_geojson, quickfix_multipolygon_shapely
from brainseg.misc.manual_correction import process_pial_gm_manual_correction, match_params
from brainseg.misc.points import transfer_points
from brainseg.parser import parse_dict_param
from brainseg.path import build_path_histo, build_path_mri
from brainseg.utils import read_histo, write_histo, get_processing_type, read_txt, hash_file, \
    extract_classification_name, calculate_name
from brainseg.viz.draw import draw_geojson_on_image


def transform_forward_histo_mri(geo, args, slide_id, processing_type):
    if processing_type == "right":
        subdir = "right"
    elif processing_type == "left":
        subdir = "left"
    elif processing_type == "both":
        subdir = "."
    else:
        raise ValueError(f"Unrecognized processing_type {processing_type}")

    transform_path = args.transforms_dir / (str(slide_id).zfill(3) + "_forward") / subdir / "TransformParameters.1.txt"
    print(transform_path, transform_path.exists())
    transform_path = str(transform_path)
    point_list = record_point_coordinates(geo)

    transferred_point_list = transfer_points(point_list, transform_path)
    dict_point_transfer = {point: transfer for point, transfer in zip(point_list, transferred_point_list)}
    transformed_geo = transform_from_dict(geo, dict_point_transfer)

    return transformed_geo


def manual_correction(histo_space, pial, params):
    # here manage the multipolygons !
    # No contained to manual correct all stuff
    polygons = flatten_geojson(histo_space)
    full_params_feats = match_params(polygons,
                                     pial, params, contained=None, warn=True)
    histo_space = transform_from_manual_correction(explode_multipolygon_geojson(histo_space),
                                                   full_params_feats, direction="forward")
    return histo_space


def run_slice(args, slice_id, dict_affine_params):
    processing_type = get_processing_type(args.schedule_steps, args.schedule_transfer_type, slice_id)
    print("Processing type", processing_type)
    histo_file = build_path_histo(args.annotations_dir, slice_id, args.merged_annotations_mask)
    histo = read_histo(histo_file)  # geojson handled for now
    histo = quickfix_multipolygon_shapely(histo)
    histo_geojson = gpd.read_file(histo_file)
    calculate_name(histo_geojson)
    # histo_geojson["name"] = histo_geojson["classification"].apply(extract_classification_name)
    # here compute manual correction
    ordered_pial, _, params, _, _, _ = process_pial_gm_manual_correction(
        dict_affine_params, histo_geojson, slice_id)
    histo_corrected = manual_correction(histo, ordered_pial, params)
    histo_resized = transform_forward_histo(histo_corrected, args)

    mri_slice_space = transform_forward_histo_mri(histo_resized, args, slice_id, processing_type)

    output_file = build_path_histo(args.mri_projections_dir, slice_id, args.merged_annotations_mask)
    write_histo(mri_slice_space, output_file)

    output_image = str(args.mri_projections_dir / f"annotation_on_mri_{slice_id}.png")
    image = imread(build_path_mri(args.mri_sections_dir, slice_id, "raw"))
    """draw_geojson_on_image(np.zeros((1000, 1000, 3)), transform_rescale(0.02, histo),
                          str(args.mri_projections_dir / f"ab_before_cor_{slice_id}.png"))
    draw_geojson_on_image(np.zeros((1000, 1000, 3)), transform_rescale(0.02, histo_corrected),
                          str(args.mri_projections_dir / f"ab_after_cor_{slice_id}.png"))"""
    draw_geojson_on_image(image, mri_slice_space, output_image)
    print("end")


def main(args):
    param_data = read_txt(args.manual_correction_file)
    hash_param = hash_file(args.manual_correction_file)
    dict_affine_params = parse_dict_param(",".join(param_data))

    for i in range(args.start, args.end, args.step):
        section_id = str(i).zfill(3)
        try:
            saved_hash = read_txt(args.transforms_dir / (section_id + "_forward") / "hash.txt")[0]
            if not saved_hash == hash_param:
                print(saved_hash, hash_param)
                warnings.warn("WARN : hash mismatch, you may have changed the manual correction params")
            run_slice(args, i, dict_affine_params)
        except (FileExistsError, FileNotFoundError) as e:
            print(str(e))
        except Exception as e:
            print("=" * 40)
            traceback.print_exc()
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
    parser.add_argument("--mri_sections_dir", type=Path, default=None)
    parser.add_argument("--manual_correction_file", type=Path, default=None)
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--schedule_steps", type=str, default=None)
    parser.add_argument("--schedule_transfer_type", type=str, default=None)
    args_ = fill_with_config(parser)

    if not args_.mri_projections_dir.exists():
        args_.mri_projections_dir.mkdir(parents=True, exist_ok=True)

    main(args_)
