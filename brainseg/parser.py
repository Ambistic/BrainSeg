import numpy as np


def parse_nomenclature(code):
    parts = code.strip().split('_')

    # Extract slice number
    slice_number = int(parts[0])

    # Extract piece number
    piece_number = int(parts[1])

    # Extract flip indicator
    flip = True if parts[2] == 'F' else False
    rotation_angle = int(parts[3])
    shift_x = int(parts[4])
    shift_y = int(parts[5])

    return slice_number, piece_number, flip, rotation_angle, shift_x, shift_y


def parse_dict_param(string):
    string = string.replace("\n", ",")
    codes = string.split(",")

    d = dict()
    for code in codes:
        if not code.strip():
            continue
        slice_number, piece_number, flip, rotation_angle, shift_x, shift_y = parse_nomenclature(code)
        d[(slice_number, piece_number)] = (flip, rotation_angle, shift_x, shift_y)

    return d


def get_arrows_from_root(root):
    return list(root.find("Elements").findall("Arrow"))


def get_arrow_coords(arrow):
    x1 = float(arrow.find("Geometry").find("X1").text)
    x2 = float(arrow.find("Geometry").find("X2").text)
    y1 = float(arrow.find("Geometry").find("Y1").text)
    y2 = float(arrow.find("Geometry").find("Y2").text)

    return np.array([[x1, y1], [x2, y2]])
