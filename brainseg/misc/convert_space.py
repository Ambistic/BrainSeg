import numpy as np


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


def pixel_slice_to_mri_3d(x, y, slice_id, angles):
    """Returns a numpy ndarray of size 3 (x, y, z)"""
    transfer_matrix = build_coord_from_param(angles[0], angles[1], angles[2], slice_id)
    point = np.array([x, y, 1])

    return transfer_matrix.T @ point


def mri_3d_to_pixel_slice(x, y, z, angles):
    """Returns a tuple of size 3, (px, py, slide_id)"""
    point = np.array([x, y, z])

    unit_transfer_matrix = build_coord_from_param(angles[0], angles[1], angles[2], 1)
    inverse_unit_transfer_matrix = np.linalg.inv(unit_transfer_matrix)
    slice_estimation = point @ inverse_unit_transfer_matrix
    slice_id = np.round(slice_estimation[2])

    transfer_matrix = build_coord_from_param(angles[0], angles[1], angles[2], slice_id)
    inverse_transfer_matrix = np.linalg.inv(transfer_matrix)
    coord_estimation = point @ inverse_transfer_matrix

    return coord_estimation[0], coord_estimation[1], slice_id