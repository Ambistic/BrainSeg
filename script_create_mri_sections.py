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


def main(args):
    d_outputs = dict(
        gm="ribbon_both_GM.nii.gz",
        wm="ribbon_both_WM.nii.gz",
        pial="ribbon_both_PIAL.nii.gz",
        raw="T1w_acpc_dc_restore.nii.gz",
    )
    for i in tqdm.tqdm(np.arange(args.start, args.end, args.step)):
        i_mri = (i - args.translation_y) / args.scale_y
        i_histo = str(int(i)).zfill(3)
        coord_values = get_values_from_wb(args.angle_x, 0, args.angle_z, i_mri)
        for ftype, fname in d_outputs.items():
            output_path = args.mri_section_dir / f"{ftype}_{i_histo}.png"

            volume_name = args.mri_dir / fname
            raw_cmd = args.wb_binary
            max_val = 256 if ftype == "raw" else 1
            cmd = f'{raw_cmd} -volume-capture-plane "{volume_name}" 1 TRILINEAR 1000 1000 ' \
                  f'0 {max_val} {coord_values} "{output_path}"'
            # print(cmd.split(" "))
            os.system(cmd)
            # subprocess.check_output(cmd.split(" "))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mri_dir", type=Path, default=None)
    parser.add_argument("-c", "--config", type=Path, default=None)
    parser.add_argument("-x", "--angle_x", type=float, default=None)
    parser.add_argument("-z", "--angle_z", type=float, default=None)
    parser.add_argument("--translation_y", type=float, default=None)
    parser.add_argument("--scale_y", type=float, default=None)
    parser.add_argument("--start", type=float, default=None)
    parser.add_argument("--end", type=float, default=None)
    parser.add_argument("--step", type=float, default=None)
    parser.add_argument("-o", "--mri_section_dir", type=Path, default=None)
    parser.add_argument("-b", "--wb_binary", type=Path, help="The path to the `wb_command` binary", default=None)
    args_ = fill_with_config(parser)
    args_.mri_section_dir.mkdir(parents=True, exist_ok=True)
    main(args_)
