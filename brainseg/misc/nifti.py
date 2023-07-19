import nibabel as nib
import numpy as np


def load_nifti(f):
    return nib.load(f)


def get_nifti_shape(nifti):
    return nifti.get_fdata().shape


def get_nifti_start_coord(nifti):
    vec = nifti.affine @ np.array([0, 0, 0, 1])
    return vec[:3]


def get_nifti_end_coord(nifti):
    vec = nifti.affine @ np.array(list(get_nifti_shape(nifti)) + [1])
    return vec[:3]
