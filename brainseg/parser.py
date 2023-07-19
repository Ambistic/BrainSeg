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
    codes = string.split(",")

    d = dict()
    for code in codes:
        slice_number, piece_number, flip, rotation_angle, shift_x, shift_y = parse_nomenclature(code)
        d[(slice_number, piece_number)] = (flip, rotation_angle, shift_x, shift_y)

    return d
