import argparse
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import geopandas as gpd
from scipy.sparse import csr_matrix
from shapely.geometry import MultiPoint, Point

import matplotlib.pyplot as plt

from brainseg.config import fill_with_config
from brainseg.misc.convert_space import pixel_slice_to_mri_3d
from brainseg.misc.nifti import load_nifti, get_nifti_end_coord, get_nifti_start_coord, get_nifti_shape
from brainseg.misc.volume_ops import interpolate, create_3d_histogram
from brainseg.path import build_path_histo
from brainseg.utils import extract_classification_name, read_txt


def build_interpolation(args, slices_indices, list_arrays):
    interp_func = interpolate(slices_indices, list_arrays)
    return [interp_func(np.round(i, 3)) for i in np.arange(
        slices_indices[0], slices_indices[-1], args.interpolation_step)]


def coordinates_to_index(point, mri_res=0.5, index_zero=np.array([96, 128, 96])):
    return point / mri_res + index_zero


def extract_points(args, slice_id):
    array_point = []
    array_name = []
    mri_index = histo_slice_to_mri_slice(args, slice_id)
    file_path = build_path_histo(args.mri_projections_dir, slice_id, args.merged_annotations_mask)
    if not Path(file_path).exists():
        return None

    obj = gpd.read_file(file_path)
    obj["name"] = obj["classification"].apply(extract_classification_name)

    for geometry, name in zip(obj["geometry"], obj["name"]):
        if isinstance(geometry, MultiPoint):
            for p in geometry.geoms:
                point = pixel_slice_to_mri_3d(p.x, p.y, mri_index, (args.angle_x, 0, args.angle_z))
                array_point.append(coordinates_to_index(point))
                array_name.append(name)
        elif isinstance(geometry, Point):
            point = pixel_slice_to_mri_3d(geometry.x, geometry.y, mri_index, (args.angle_x, 0, args.angle_z))
            array_point.append(coordinates_to_index(point))
            array_name.append(name)
        else:
            continue

    if len(array_point) == 0:
        return None

    return array_point, array_name


def bin_slice_annotation(array):
    array = array.round().astype(int)
    size = 1000
    if array.size == 0:
        return csr_matrix(np.zeros((1000, 1000)))
    hist, _, _ = np.histogram2d(array[:, 0], array[:, 1],
                                bins=(size, size), range=((0, size), (0, size)))

    hist = csr_matrix(hist)
    return hist


def histo_slice_to_mri_slice(args, index):
    # mri_id = (section_id - translation) / scale
    mri_index = (index - args.translation_y) / args.scale_y
    return mri_index


def bin_mri_coord(args, slices_indices, list_arrays, ref_nifti):
    new_slice_indices = [np.round(i, 3) for i in np.arange(
        slices_indices[0], slices_indices[-1], args.interpolation_step
    )]
    # for each coord, transfer to raw mri coords
    full_list_points = []
    weights = []
    for index, sparse_array in zip(new_slice_indices, list_arrays):
        mri_index = histo_slice_to_mri_slice(args, index)
        coo = sparse_array.tocoo()
        # TODO here the count is not counted
        for x, y, w in zip(coo.row, coo.col, coo.data):
            full_list_points.append(
                pixel_slice_to_mri_3d(x, y, mri_index, (args.angle_x, 0, args.angle_z))
            )
            weights.append(w)

    # (N, 3) array
    full_list_points = np.array(full_list_points)
    # round the coords
    # full_list_points = full_list_points.round().astype(int)
    shape = get_nifti_shape(ref_nifti)
    bin_starts = get_nifti_start_coord(ref_nifti)
    bin_stops = get_nifti_end_coord(ref_nifti)
    hist = create_3d_histogram(full_list_points, shape, bin_starts, bin_stops, weights=weights)
    return hist


def main(args):
    excluded = list(map(lambda x: int(x.strip()), read_txt(args.exclude_file)))
    slices_indices = list(range(args.start, args.end, args.step))
    # slices_indices = [129, 130, 131]
    ref_nifti = load_nifti(args.nifti_reference)
    list_slice = [extract_points(args, slice_id) if slice_id not in excluded else None
                  for slice_id in slices_indices]
    slices_indices, list_slice = zip(*filter(lambda x: x[1] is not None, zip(slices_indices, list_slice)))
    coords, names = list(zip(*list_slice))
    coords = sum(coords, [])
    names = sum(names, [])
    for c, n in zip(coords, names):
        print(n, c)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=Path, default=Path("/media/tower/LaCie/REGISTRATION_PROJECT/"
                                                                  "TMP_PIPELINE_M148/config.ini"))
    # parser.add_argument("--mri_voxel_size_mm", type=float, default=None)
    parser.add_argument("--merged_annotations_mask", type=str, default=None)
    parser.add_argument("--interpolation_step", type=float, default=None)
    parser.add_argument("--nifti_reference", type=str, default=None)
    parser.add_argument("--mri_projections_dir", type=Path, default=None)
    parser.add_argument("--cell_types", type=str, default=None)
    parser.add_argument("--exclude_file", type=str, default=None)
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--translation_y", type=int, default=None)
    parser.add_argument("--scale_y", type=float, default=None)
    parser.add_argument("--angle_x", type=int, default=None)
    parser.add_argument("--angle_z", type=int, default=None)
    parser.add_argument("--internal_surface", type=str, default=None)
    parser.add_argument("--mid_surface", type=str, default=None)
    parser.add_argument("--external_surface", type=str, default=None)
    parser.add_argument("--wb_binary", type=str, default=None)
    args_ = fill_with_config(parser)
    args_.cell_types = list(map(str.strip, args_.cell_types.split(",")))

    main(args_)
