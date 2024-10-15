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
    os.system(f'bash script_process_mri.sh "{args.mri_dir.absolute()}" "{args.wb_binary.parent.absolute()}"')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=Path, default=Path("/media/tower/LaCie/REGISTRATION_PROJECT/"
                                                                  "TMP_PIPELINE_M148/config.ini"))
    parser.add_argument("--mri_dir", type=Path, default=None)
    parser.add_argument("--wb_binary", type=Path, default=None)

    args_ = fill_with_config(parser)

    main(args_)
