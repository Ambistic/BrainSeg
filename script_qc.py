import argparse
import configparser
import os.path
from pathlib import Path

import pandas as pd

from brainseg.parser import parse_dict_param
from brainseg.path import build_path_histo
from brainseg.utils import read_txt, hash_file


def test_int(x):
    try:
        int(x)
    except:
        res = False
    else:
        res = True
    return res


def test_float(x):
    try:
        float(x)
    except:
        res = False
    else:
        res = True
    return res


def test_not_fill(x: str):
    return "<FILL>" not in x


def test_path_exists(x):
    return os.path.exists(x)


MAP_CHECKS = {
    "int": (test_int, "%s is not an integer : %s"),
    "float": (test_float, "%s is not a float : %s"),
    "path_exists": (test_path_exists, "%s path does not exists : %s"),
    "not_fill": (test_not_fill, "%s has not been filled : %s"),
}

# divide in critical and non critical ?
LIST_CHECKS = [
    ("angle_x", "float"),
    ("angle_z", "float"),
    ("translation_y", "float"),
    ("scale_y", "float"),
    ("interpolation_step", "float"),
    ("mri_voxel_size_mm", "float"),
    ("mpp_plotfast", "float"),
    ("slice_thickness_mm", "float"),
    ("inter_histo_slice_mm", "float"),

    ("start", "int"),
    ("end", "int"),
    ("step", "int"),
    ("histo_downscale", "int"),

    ("cell_types", "not_fill"),
    ("schedule_steps", "not_fill"),
    ("hemisphere_surface", "not_fill"),

    ("root", "path_exists"),
    ("nifti_reference", "path_exists"),
    ("internal_surface", "path_exists"),
    ("mid_surface", "path_exists"),
    ("external_surface", "path_exists"),

    ("annotations_mask", "not_fill"),
    ("full_annotations_mask", "not_fill"),
    ("slides_mask", "not_fill"),
    ("histo_mask", "not_fill"),
    ("fluo_mask", "not_fill"),
    ("plotfast_mask", "not_fill"),
    ("merged_annotations_mask", "not_fill"),

    ("mri_sections_dir", "path_exists"),
    ("mri_projections_dir", "path_exists"),
    ("transforms_dir", "path_exists"),
    ("annotations_dir", "path_exists"),
    ("slides_dir", "path_exists"),
    ("histo_dir", "path_exists"),
    ("fluo_dir", "path_exists"),
    ("plotfast_dir", "path_exists"),
    ("mri_dir", "path_exists"),
    ("mri_brain_file", "path_exists"),
    ("mri_gm_file", "path_exists"),
    ("mri_section_dir", "path_exists"),
    ("mri_atlas_dir", "path_exists"),
    ("manual_correction_file", "path_exists"),
    ("exclude_file", "path_exists"),
    ("wb_binary", "path_exists"),
    ("segmentation_weights", "path_exists"),
    ("siamese_weights", "path_exists"),
]


def check_config_file(args):
    print("== Begin of config check ==")
    config = configparser.ConfigParser()
    config.read(args.config)
    for name, type_check in LIST_CHECKS:
        func_check, msg_check = MAP_CHECKS[type_check]
        value = config["DEFAULT"].get(name, "")
        if not func_check(value):
            print("->", msg_check % (name, value))
            print()
    print("== End of config check ==")


def build_report_csv(args):
    config = configparser.ConfigParser()
    config.read(args.config)
    param = config["DEFAULT"]
    indexes = list(range(int(param["start"]), int(param["end"]), int(param["step"])))
    indexes.append("Total")
    df = pd.DataFrame(index=indexes)

    for col_name, dir_name, mask_name in [
        ("czi", "slides_dir", "slides_mask"),
        ("fluo_svg", "plotfast_dir", "plotfast_mask"),
        ("fluo_geojson", "fluo_dir", "fluo_mask"),
        ("png", "histo_dir", "histo_mask"),
        ("mri_sections", "mri_sections_dir", "raw_sections_mask"),
        ("segmentation", "annotations_dir", "full_annotations_mask"),
        ("merged_annotation", "annotations_dir", "merged_annotations_mask"),
    ]:
        ls = list(map(
            lambda x: "X" if x else "",
            [test_path_exists(build_path_histo(param.get(dir_name, ""), section_id, param.get(mask_name, "%s")))
             for section_id in indexes[:-1]]
        ))
        ls.append(len(list(filter(lambda x: x, ls))))
        df[col_name] = ls

    for col_name, dir_name, mask_name in [
        ("transforms", "transforms_dir", "%s_forward/TransformParameters.1.txt"),
        ("atlas", "mri_atlas_dir", "atlas_%s.png"),
    ]:
        ls = list(map(
            lambda x: "X" if x else "",
            [test_path_exists(Path(param.get(dir_name, "")) / (mask_name % str(section_id).zfill(3)))
             for section_id in indexes[:-1]]
        ))
        ls.append(len(list(filter(lambda x: x, ls))))
        df[col_name] = ls

    excluded = list(map(lambda x: int(x.strip()), read_txt(param.get("exclude_file"))))
    ls_excluded = list(map(
            lambda x: "X" if x else "",
            [section_id in excluded for section_id in indexes[:-1]]
        ))
    ls_excluded.append(len(list(filter(lambda x: x, ls_excluded))))
    df["excluded"] = ls_excluded

    param_data = read_txt(param.get("manual_correction_file"))
    dict_affine_params = parse_dict_param(",".join(param_data))
    manually_corrected = list(set([x for x, _ in dict_affine_params.keys()]))
    ls_corrected = list(map(
        lambda x: "X" if x else "",
        [section_id in manually_corrected for section_id in indexes[:-1]]
    ))
    ls_corrected.append(len(list(filter(lambda x: x, ls_corrected))))
    df["corrected"] = ls_corrected

    df["data_ok"] = (df["czi"].astype(bool) & df["png"].astype(bool) & df["mri_sections"].astype(bool)
                     & (df["fluo_svg"].astype(bool) | df["fluo_geojson"].astype(bool))
                     ).apply(lambda x: "X" if x else "")
    df.loc["Total", "data_ok"] = len(list(filter(lambda x: x, list(df["data_ok"])[:-1])))
    df.to_csv(args.config.parent / "summary.csv")
    """
    2) check all the indexed data (czi, png, svg, geojson, mri)
    3) check intermediary data (seg, merged, sections, transforms, atlas)
    4) review the checks ? how ?
    5) add the exclude.txt and manual_correction.txt info
    """


def main(args):
    check_config_file(args)
    build_report_csv(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=Path)
    args_ = parser.parse_args()
    main(args_)
