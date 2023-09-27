import numpy as np
from skimage.transform import rotate

from brainseg.image import resize_and_pad_center
from brainseg.polygon import rescale_polygon, translate_to_origin
from brainseg.viz.draw import draw_polygon


def fill_subpart_from_mask(big_image, subpart_image, mask, origin_coords):
    x, y = origin_coords
    subpart_width, subpart_height = subpart_image.shape[:2]
    big_width, big_height = big_image.shape[:2]

    # Calculate the region that can be copied from the big image to the subpart
    min_x = max(x, 0)
    min_y = max(y, 0)
    max_x = min(x + subpart_width, big_width)
    max_y = min(y + subpart_height, big_height)

    # Calculate the offset of the region in the big image
    x_offset = max(0, -x)
    y_offset = max(0, -y)

    # Calculate the corresponding region in the subpart
    subpart_region = subpart_image[x_offset:x_offset+(max_x-min_x), y_offset:y_offset+(max_y-min_y)]
    # print(locals())

    # Copy the valid region from the big image to the subpart image where the mask is True
    valid_mask = mask[x_offset:x_offset+(max_x-min_x), y_offset:y_offset+(max_y-min_y)]
    big_image[min_x:max_x, min_y:max_y][valid_mask] = subpart_region[valid_mask]

    return big_image


def image_manual_correction(image, params, polygons, background=0, scale=1., swap_xy=False, margin=(0, 0)):
    if swap_xy:
        image = image.transpose((1, 0, 2))

    sh = image.shape
    res_image = np.zeros((sh[0] + margin[0], sh[1] + margin[1], sh[2]), dtype=np.uint8)
    if background != 0:
        res_image.fill(background)

    for polygon, param in zip(polygons, params):
        poly = rescale_polygon(polygon, scale)
        minx, miny, maxx, maxy = map(int, poly.bounds)

        if param is None:
            flip, rotation_angle, shift_x, shift_y = False, 0, 0, 0
            center_x, center_y = (minx + maxx) / 2, (miny + maxy) / 2
        else:
            center_x, center_y, flip, rotation_angle, shift_x, shift_y = param
            center_x, center_y = center_x * scale, center_y * scale
        shift_x, shift_y = shift_x * scale, shift_y * scale
        # center_x, center_y = poly.centroid.coords[0]
        # center_x, center_y = (minx + maxx) / 2, (miny + maxy) / 2
        width, height = maxx - minx, maxy - miny
        size = int(1.5 * max(width, height))
        half_size = int(size / 2)

        subimage = image[minx:maxx, miny:maxy]
        subimage_mask = np.zeros(subimage.shape[:2], dtype=bool)
        subimage_mask = draw_polygon(subimage_mask, translate_to_origin(poly))

        # resize
        subimage_mask = resize_and_pad_center(subimage_mask, size, size)
        subimage = resize_and_pad_center(subimage, size, size)

        if flip:
            subimage_mask = subimage_mask[::-1]
            subimage = subimage[::-1]

        if rotation_angle != 0:
            subimage_mask = rotate(subimage_mask, rotation_angle)
            subimage = rotate(subimage, rotation_angle)

        subimage_mask = subimage_mask.astype(bool)

        fill_subpart_from_mask(res_image, subimage, subimage_mask,
                               (int(center_x - half_size + shift_x), int(center_y - half_size + shift_y)))

    if swap_xy:
        res_image = res_image.transpose((1, 0, 2))

    return res_image
