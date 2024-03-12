import tempfile

import itk
import numpy as np


def get_affine_matrix_from_elastix(image_histo, image_mri):
    fixed = itk.GetImageFromArray(image_histo.astype(np.float32), is_vector=False)
    moving = itk.GetImageFromArray(image_mri.astype(np.float32), is_vector=False)

    parameter_object = itk.ParameterObject.New()
    parameter_map_affine = parameter_object.GetDefaultParameterMap('affine')

    parameter_map_affine["Transform"] = ['EulerTransform']
    parameter_map_affine["MaximumNumberOfIterations"] = ['1024']
    parameter_map_affine["NumberOfResolutions"] = ['6.000000']  # could be 5
    parameter_map_affine["Metric"] = ["AdvancedNormalizedCorrelation"]
    parameter_map_affine["NumberOfSpatialSamples"] = ["50000"]

    # parameter_map_affine["AutomaticTransformInitialization"] = ["true"]
    # parameter_map_affine["AutomaticTransformInitializationMethod"] = ["CenterOfGravity"]

    parameter_object.AddParameterMap(parameter_map_affine)

    # Call registration function and specify output directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        result_image, result_transform_parameters = itk.elastix_registration_method(
            fixed, moving,
            parameter_object=parameter_object,
            output_directory=str(tmp_dir))

    return (
        result_transform_parameters.GetParameter(0, "TransformParameters"),
        result_transform_parameters.GetParameter(0, "CenterOfRotationPoint"),
        np.array(result_image),
    )


def get_affine_transform_matrix(fixed, moving):
    """
    Fixed and moving such that moving = fixed @ affine_matrix
    In other words, the moving is the result of the transformation
    of the fixed image with the returned matrix
    """
    assert isinstance(fixed, np.ndarray)
    assert isinstance(moving, np.ndarray)
    print(fixed.shape, moving.shape)
    # assert fixed.shape == moving.shape

    sx, sy = fixed.shape
    mat, center, res = get_affine_matrix_from_elastix(moving, fixed)  # maybe the opposite
    processed_mat = map(float, mat)
    # s, theta, U, V = processed_mat
    theta, U, V = processed_mat
    cx, cy = map(float, center)

    # u = V - (sy / 2 * b - sx / 2 * (1 - a))
    # v = U - (sx / 2 * c - sy / 2 * (1 - d))

    # print(f"\n ==== S = {s} ==== \n")

    a, b, c, d = np.cos(theta), np.sin(theta), -np.sin(theta), np.cos(theta)
    u = V - (cx * b - cy * (1 - a))
    v = U - (cy * c - cx * (1 - d))

    return np.array([[a, b, u], [c, d, v]]), res
