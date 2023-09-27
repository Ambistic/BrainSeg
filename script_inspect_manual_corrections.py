import argparse

from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
from shapely.errors import ShapelyDeprecationWarning
from skimage import io
import geopandas as gpd
from skimage.transform import rescale

from brainseg.config import fill_with_config
from brainseg.geo import fix_geojson_file
from brainseg.misc.image_geometry import image_manual_correction
from brainseg.misc.manual_correction import process_pial_gm_manual_correction
from brainseg.parser import parse_dict_param
from brainseg.path import build_path_mri, build_path_histo
from brainseg.polygon import validate_no_overflow, \
    transform_polygons
from brainseg.utils import extract_classification_name, read_txt
from brainseg.viz.draw import draw_polygons

import warnings
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)


def export_correction(
        args,
        dir_histo, dir_mri, dir_histo_annotation,
        filename_mask_raw, filename_mask_annotation, section_id,
        dict_affine_params
):
    mri_raw = io.imread(build_path_mri(dir_mri, section_id, "raw"))
    histo_raw = io.imread(build_path_histo(dir_histo, section_id, filename_mask_raw))
    histo_raw = rescale(histo_raw, args.scale_inspect / 0.1, channel_axis=2,
                        preserve_range=True, anti_aliasing=False)
    histo_raw = histo_raw.astype(int)
    # build polygons
    histo_path = build_path_histo(
        dir_histo_annotation, section_id, filename_mask_annotation)
    fix_geojson_file(histo_path)
    histo_geojson = gpd.read_file(histo_path)
    histo_geojson["name"] = histo_geojson["classification"].apply(extract_classification_name)

    ordered_polygons, _, params, _, transformed_polygons, transformed_gm = process_pial_gm_manual_correction(
        dict_affine_params, histo_geojson, section_id)

    print(params)
    validate_no_overflow(transformed_polygons, raises=False)

    histo_mod = image_manual_correction(histo_raw, params, ordered_polygons, margin=(int(200 / 2.5), int(100 / 2.5)),
                                        swap_xy=True, background=0, scale=args.scale_inspect)

    export_multi_plot(args, histo_mod, histo_raw, mri_raw, ordered_polygons, section_id, transformed_polygons)


def export_multi_plot(args, histo_mod, histo_raw, mri_raw, ordered_polygons, section_id, transformed_polygons):
    arr = np.zeros((600, 600))
    arr_ori = draw_polygons(arr, ordered_polygons, rescale=args.scale_inspect, color_iteration=True)
    arr_trs = draw_polygons(arr, transformed_polygons, rescale=args.scale_inspect, color_iteration=True)

    def format_func(value, tick_number):
        return int(value / args.scale_inspect)

    fig, axes = plt.subplots(3, 2, figsize=(16, 12), dpi=80)

    # Set the custom tick formatter for all subplots
    for ax in [x for y in axes for x in y]:
        ax.xaxis.set_major_formatter(FuncFormatter(format_func))
        ax.yaxis.set_major_formatter(FuncFormatter(format_func))

    axes[0, 0].imshow(arr_ori.T)
    axes[0, 1].imshow(arr_trs.T)
    axes[1, 0].imshow(mri_raw)
    axes[2, 0].imshow(histo_raw)
    axes[2, 1].imshow(histo_mod)
    fig.savefig(args.transforms_dir / f"{section_id}_correction.png")


def create_transform_from_dirs(args, dir_histo, dir_mri, dir_histo_annotation,
                               filename_mask_raw, filename_mask_annotation,
                               output_dir, start=0, end=500, step=1):
    sections_made = []
    param_data = read_txt(args.manual_correction_file)
    dict_affine_params = parse_dict_param(",".join(param_data))
    section_with_param = list(set([x[0] for x in dict_affine_params.keys()]))

    if args.single != -1 and args.single in section_with_param:
        export_correction(
            args,
            dir_histo, dir_mri, dir_histo_annotation,
            filename_mask_raw, filename_mask_annotation, args.single,
            dict_affine_params,
        )
        return

    for section_id in range(start, end, step):
        if section_id in section_with_param:
            print("=" * 40)
            print("Running for section", section_id)
            export_correction(
                args,
                dir_histo, dir_mri, dir_histo_annotation,
                filename_mask_raw, filename_mask_annotation, section_id,
                dict_affine_params,
            )


def main(args):
    create_transform_from_dirs(args, args.histo_dir, args.mri_section_dir, args.annotations_dir,
                               args.histo_mask, args.full_annotations_mask,
                               args.transforms_dir,
                               start=args.start, end=args.end, step=args.step)


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
    parser.add_argument("--manual_correction_file", type=Path, default=None)
    parser.add_argument("--hemisphere", type=str, default=None)
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--scale_inspect", type=float, default=0.01)
    parser.add_argument("--single", type=int, default=-1)

    args_ = fill_with_config(parser)
    main(args_)
