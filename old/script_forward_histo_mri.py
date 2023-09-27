import argparse
from pathlib import Path

from skimage.io import imread

from brainseg.config import fill_with_config
from brainseg.geo import record_point_coordinates, transform_from_dict, transform_forward_histo
from brainseg.misc.points import transfer_points
from brainseg.path import build_path_histo, build_path_mri
from brainseg.utils import read_histo, write_histo
from brainseg.viz.draw import draw_geojson_on_image


def transform_forward_histo_mri(geo, args, slide_id):
    transform_path = args.transforms_dir / (str(slide_id).zfill(3) + "_forward") / "TransformParameters.1.txt"
    print(transform_path, transform_path.exists())
    transform_path = str(transform_path)
    point_list = record_point_coordinates(geo)

    transferred_point_list = transfer_points(point_list, transform_path)
    dict_point_transfer = {point: transfer for point, transfer in zip(point_list, transferred_point_list)}
    transformed_geo = transform_from_dict(geo, dict_point_transfer)

    return transformed_geo


def run_slice(args, slice_id):
    histo_file = build_path_histo(args.annotations_dir, slice_id, args.full_annotations_mask)
    histo = read_histo(histo_file)  # geojson handled for now
    histo_resized = transform_forward_histo(histo, args)

    mri_slice_space = transform_forward_histo_mri(histo_resized, args, slice_id)

    output_file = build_path_histo(args.mri_projections_dir, slice_id, args.full_annotations_mask)
    write_histo(mri_slice_space, output_file)

    output_image = str(args.mri_projections_dir / f"annotation_on_mri_{slice_id}.png")
    image = imread(build_path_mri(args.mri_sections_dir, slice_id, "raw"))
    draw_geojson_on_image(image, mri_slice_space, output_image)
    print("end")


def main(args):
    for i in range(args.start, args.end, args.step):
        run_slice(args, i)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=Path, default=Path("/media/tower/LaCie/REGISTRATION_PROJECT/"
                                                                  "TMP_PIPELINE_M148/config.ini"))
    parser.add_argument("--histo_downscale", type=float, default=None)
    parser.add_argument("--full_annotations_mask", type=str, default=None)
    parser.add_argument("--annotations_dir", type=Path, default=None)
    parser.add_argument("--transforms_dir", type=Path, default=None)
    parser.add_argument("--mri_projections_dir", type=Path, default=None)
    parser.add_argument("--mri_sections_dir", type=Path, default=None)
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--step", type=int, default=None)
    args_ = fill_with_config(parser)

    main(args_)
