import argparse
import os
from pathlib import Path
import numpy as np
import subprocess
import configparser
import tqdm

from brainseg.config import fill_with_config


def build_coord_from_param(ox, oy, oz, slice_number):
    xbounds = np.array([-32, 32])
    zbounds = np.array([28, -22])
    interpcoord = np.array([0, slice_number, 0])
    pixelsize = (zbounds[0] - zbounds[1]) / 999
    radianx = ox * np.pi / 180
    radiany = oy * np.pi / 180
    radianz = oz * np.pi / 180

    rotatematx = np.array([
        [1, 0, 0],
        [0, np.cos(radianx), -np.sin(radianx)],
        [0, np.sin(radianx), np.cos(radianx)]
    ])
    rotatematy = np.array([
        [np.cos(radiany), 0, np.sin(radiany)],
        [0, 1, 0],
        [-np.sin(radiany), 0, np.cos(radiany)]
    ])
    rotatematz = np.array([
        [np.cos(radianz), -np.sin(radianz), 0],
        [np.sin(radianz), np.cos(radianz), 0],
        [0, 0, 1],
    ])
    rotatemat = rotatematz @ rotatematy @ rotatematx
    normal = rotatemat @ np.array([0, 1, 0]).T

    # return rotatemat

    planeconst = interpcoord @ normal / normal[1]
    centercoord = np.array([
        xbounds.mean(),
        planeconst - xbounds.mean() * normal[0] / normal[1] - zbounds.mean() * normal[2] / normal[1],
        zbounds.mean(),
    ])

    zspacing = rotatemat @ np.array([0, 0, -1]).T * pixelsize
    xspacing = rotatemat @ np.array([1, 0, 0]).T * pixelsize
    topleft = centercoord.T - 500 * xspacing - 500 * zspacing

    return np.array([xspacing, zspacing, topleft])


def get_values_from_wb(ox, oy, oz, slice_number):
    pixel2mm = build_coord_from_param(ox, oy, oz, slice_number)

    blmm = pixel2mm.T @ np.array([1, 1000, 1])
    brmm = pixel2mm.T @ np.array([1000, 1000, 1])
    tlmm = pixel2mm.T @ np.array([1, 1, 1])

    values = list(np.concatenate([blmm, brmm, tlmm]))
    text = " ".join(map(lambda x: f"{x:.2f}", values))
    return text


def generate_reduced_mapping(args):
    ATLAS_ROI = args.mri_dir / "atlas_roi.func.gii"
    SMOOTH_ATLAS_ROI = args.mri_dir / "smooth_atlas_roi.func.gii"
    OUT_MAPPING = args.mri_dir / "out_mapping_h.nii.gz"
    MASK_REDUCE = args.mri_dir / "mask_mapping.nii.gz"
    LAYER4_VOL = args.mri_dir / "ribbonSpace.nii.gz"

    print("Generating ROIs")
    os.system(f"{args.wb_binary} -gifti-all-labels-to-rois '{args.atlas_file}' 1 '{ATLAS_ROI}'")
    os.system(f"{args.wb_binary} -metric-smoothing '{args.midthickness_surface_file}' '{ATLAS_ROI}' "
              f"'{args.atlas_smoothing}' '{SMOOTH_ATLAS_ROI}'")

    os.system(f"{args.wb_binary} -surface-cortex-layer '{args.white_surface_file}' '{args.pial_surface_file}' "
              f"'{args.location_l4}' '{args.layer4_surface_file}'")
    os.system(f"{args.wb_binary} -create-signed-distance-volume '{args.layer4_surface_file}' "
              f"'{args.volume_file}' '{LAYER4_VOL}' -exact-limit 1 -approx-limit 7")
    # then use (layer4 < 0) for infra or (layer4 > 0) for supra in the -volume-math

    os.system(
        f"{args.wb_binary} -metric-math 'round(metric * 10)' '{SMOOTH_ATLAS_ROI}' -var metric '{SMOOTH_ATLAS_ROI}'")

    print("Creating volume")
    os.system(f"{args.wb_binary} -label-to-volume-mapping '{SMOOTH_ATLAS_ROI}' '{args.midthickness_surface_file}' "
              f"'{args.volume_file}' '{OUT_MAPPING}' -ribbon-constrained '{args.white_surface_file}' "
              f"'{args.pial_surface_file}'")

    os.system(f"{args.wb_binary} -volume-reduce '{OUT_MAPPING}' INDEXMAX '{args.reduced_mapping_file}'")
    os.system(f"{args.wb_binary} -volume-reduce '{OUT_MAPPING}' SUM '{MASK_REDUCE}'")
    os.system(f"{args.wb_binary} -volume-math 'round( (mask > 5) * (1 + round(val) + (120 * (layer4 > 0))) )' "
              f"'{args.reduced_mapping_file}' -var mask '{MASK_REDUCE}' "
              f"-var val '{args.reduced_mapping_file}' -var layer4 '{LAYER4_VOL}'")
    print("Reduced mapping done")


def main(args):
    if not args.skip_generation:
        print("Generating")
        generate_reduced_mapping(args)
    else:
        print("Skipping generation")

    for i in tqdm.tqdm(np.arange(args.start, args.end, args.step)):
        i_mri = (i - args.translation_y) / args.scale_y
        i_histo = str(int(i)).zfill(3)
        coord_values = get_values_from_wb(args.angle_x, 0, args.angle_z, i_mri)

        output_path = args.mri_atlas_dir / f"atlas_{i_histo}.png"

        raw_cmd = args.wb_binary
        cmd = f'{raw_cmd} -volume-capture-plane "{args.reduced_mapping_file}" 1 ENCLOSING_VOXEL 1000 1000 ' \
              f'0 255 {coord_values} "{output_path}"'  # maybe here it should be 255 instead of 256

        os.system(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=Path, default=None)
    parser.add_argument("-x", "--angle_x", type=float, default=None)
    parser.add_argument("-z", "--angle_z", type=float, default=None)
    parser.add_argument("--location_l4", type=float, default=None)
    parser.add_argument("--translation_y", type=float, default=None)
    parser.add_argument("--scale_y", type=float, default=None)
    parser.add_argument("--start", type=float, default=None)
    parser.add_argument("--end", type=float, default=None)
    parser.add_argument("--step", type=float, default=None)
    parser.add_argument("--atlas_smoothing", type=float, default=None)
    parser.add_argument("--atlas_file", type=Path, default=None)
    parser.add_argument("--volume_file", type=Path, default=None)
    parser.add_argument("--midthickness_surface_file", type=Path, default=None)
    parser.add_argument("--white_surface_file", type=Path, default=None)
    parser.add_argument("--pial_surface_file", type=Path, default=None)
    parser.add_argument("--layer4_surface_file", type=Path, default=None)
    parser.add_argument("--reduced_mapping_file", type=Path, default=None)
    parser.add_argument("--mri_atlas_dir", type=Path, default=None)
    parser.add_argument("-m", "--mri_dir", type=Path, default=None)
    parser.add_argument("-b", "--wb_binary", type=Path, help="The path to the `wb_command` binary", default=None)
    parser.add_argument("--skip_generation", action="store_true", default=False)

    args_ = fill_with_config(parser)
    args_.mri_atlas_dir.mkdir(parents=True, exist_ok=True)
    main(args_)
