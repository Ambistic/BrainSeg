import re
import tempfile
from pathlib import Path

import itk
import numpy as np


def parse_output_point_file(fname):
    data = {}
    with open(fname, 'r') as f:
        for line in f:
            if not line.startswith('Point'):
                continue

            point_data = {}
            point_id = int(re.findall(r'\d+', line)[0])
            for field in re.findall(r';.*?(\w+)\s=\s(\[.*?\])', line):
                point_data[field[0]] = [float(x) for x in re.findall(r'-?\d+\.?\d*', field[1])]
            data[point_id] = point_data
    return data


def write_coords_to_file(coords, filename):
    with open(filename, 'w') as f:
        f.write(f'index\n')
        f.write(f'{len(coords)}\n')
        for coord in coords:
            f.write(f'{coord[0]} {coord[1]}\n')


def transfer_points(point_list, transform_path):
    with tempfile.TemporaryDirectory() as tmpdir:
        filename = Path(tmpdir) / "input.txt"
        write_coords_to_file(point_list, filename)
        transform_parameters = itk.ParameterObject.New()
        transform_parameters.AddParameterFile(transform_path)

        fake_img = itk.GetImageFromArray(np.zeros((500, 500)).astype(np.float32), is_vector=False)
        # Procedural interface of transformix filter
        result_point_set = itk.transformix_pointset(
            fake_img, transform_parameters,
            fixed_point_set_file_name=str(filename),
            output_directory=tmpdir)

        points = parse_output_point_file(Path(tmpdir) / "outputpoints.txt")

        parsed_point = [points[i]["Deformation"] for i in range(len(points))]

    return parsed_point
