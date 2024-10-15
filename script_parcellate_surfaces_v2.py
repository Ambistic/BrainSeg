import argparse
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import geopandas as gpd
import pandas as pd
from scipy.sparse import csr_matrix
from shapely.geometry import MultiPoint, Point

import matplotlib.pyplot as plt

from brainseg.config import fill_with_config
from brainseg.misc.convert_space import pixel_slice_to_mri_3d
from brainseg.misc.nifti import load_nifti, get_nifti_end_coord, get_nifti_start_coord, get_nifti_shape
from brainseg.misc.volume_ops import interpolate, create_3d_histogram
from brainseg.path import build_path_histo
from brainseg.utils import extract_classification_name, read_txt


def main(args):
    """
    wb_command -cifti-create-dense-scalar $DSCALAR -left-metric $LEFT_METRIC -roi-left $ROI_LEFT -right-metric $RIGHT_METRIC -roi-right $ROI_RIGHT

    ## parcellate
    wb_command -cifti-parcellate $DSCALAR "$ATLAS" COLUMN $PSCALAR -method SUM

    ## convert to table
    wb_command -cifti-convert -to-text $PSCALAR $TXT -col-delim ' '

    ## Get the areal names only for the right hemisphere
    wb_command -cifti-label-export-table "$ATLAS" 1 $ATLAS_TABLE
    """
    for cell_type in args.cell_types:
        out_surf_name = os.path.join(args.mri_processing_dir, f"cell_density_{cell_type}.func.gii")
        left_metric = os.path.join(args.mri_processing_dir, f"cell_density_{cell_type}_L.func.gii")
        right_metric = os.path.join(args.mri_processing_dir, f"cell_density_{cell_type}_R.func.gii")
        dscalar = os.path.join(args.mri_processing_dir, f"{cell_type}_LR.dscalar.nii")
        pscalar = os.path.join(args.mri_processing_dir, f"{cell_type}_LR.pscalar.nii")
        txt_file = os.path.join(args.mri_processing_dir, f"{cell_type}_values.txt")
        atlas_table = os.path.join(args.mri_processing_dir, f"{cell_type}_atlas.txt")
        output_csv = os.path.join(args.output_dir, f"forward_{cell_type}.csv")

        os.system(f"cp '{out_surf_name}' '{left_metric}'")
        os.system(f"cp '{out_surf_name}' '{right_metric}'")
        os.system(f"'{args.wb_binary}' -set-structure '{left_metric}' CORTEX_LEFT")
        os.system(f"'{args.wb_binary}' -set-structure '{right_metric}' CORTEX_RIGHT")

        os.system(f"'{args.wb_binary}' -cifti-create-dense-scalar '{dscalar}' -left-metric '{left_metric}' "
                  f"-roi-left '{args.roi_left}' -right-metric '{right_metric}' -roi-right '{args.roi_right}'")
        os.system(f"'{args.wb_binary}' -cifti-parcellate '{dscalar}' '{args.atlas}' COLUMN '{pscalar}' -method SUM")
        os.system(f"'{args.wb_binary}' -cifti-convert -to-text '{pscalar}' '{txt_file}' -col-delim ' '")
        os.system(f"'{args.wb_binary}' -cifti-label-export-table '{args.atlas}' 1 '{atlas_table}'")

        # here python process to output a csv file
        atlas_lines = list(map(str.strip, read_txt(atlas_table)[::2]))
        values = list(map(lambda x: float(x.strip()), read_txt(txt_file)))
        df = pd.DataFrame(dict(areas=atlas_lines, values=values))
        df.to_csv(output_csv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=Path, default=Path("/media/tower/LaCie/REGISTRATION_PROJECT/"
                                                                  "TMP_PIPELINE_M148/config.ini"))
    parser.add_argument("--cell_types", type=str, default=None)
    parser.add_argument("--mri_processing_dir", type=Path, default=None)
    parser.add_argument("--roi_left", type=str, default=None)
    parser.add_argument("--roi_right", type=str, default=None)
    parser.add_argument("--nifti_reference", type=str, default=None)
    parser.add_argument("--atlas", type=str, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--wb_binary", type=str, default=None)

    args_ = fill_with_config(parser)
    args_.cell_types = list(map(str.strip, args_.cell_types.split(",")))

    if not args_.output_dir.exists():
        args_.output_dir.mkdir(parents=True, exist_ok=True)

    main(args_)
