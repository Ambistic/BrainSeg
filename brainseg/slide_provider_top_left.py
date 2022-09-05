from math import ceil
from os.path import basename

import numpy as np
from pathlib import Path
import aicspylibczi
from skimage.transform import resize
from PIL import Image

from .provider import DataHandler, provider
from .slidedata.mask import load_mask_as_image


def get_mask_from_slidepath(masks_root, slide_name, area="whitematter"):
    fp = Path(masks_root) / slide_name / (area + ".png")
    if fp.exists():
        return fp
    return None


def open_image(slide, origine, downscale, size):
    scale = 1 / downscale
    bbox = slide.get_mosaic_bounding_box()
    max_size_fit = bbox.w, bbox.h
    size_down = int(size * downscale)
    size_fit = (
        max(min(max_size_fit[0], origine[0] + size_down) - origine[0], 0),
        max(min(max_size_fit[1], origine[1] + size_down) - origine[1], 0),
    )

    delta_origine = max(0, -origine[0]), max(0, -origine[1])
    delta_origine_down = ceil(delta_origine[0] / downscale), ceil(delta_origine[1] / downscale)

    arr = np.zeros((size, size, 3))
    try:
        image = slide.read_mosaic((origine[0] + bbox.x + delta_origine[0], origine[1] + bbox.y + delta_origine[1],
                                   size_fit[0] - delta_origine[0], size_fit[1] - delta_origine[1]),
                                  C=0, scale_factor=scale)
    except Exception:
        print("failure for", slide, origine, downscale, size, max_size_fit)
        raise
    image = image.reshape(image.shape[-3:])

    # here it's reversed
    arr[delta_origine_down[1]:delta_origine_down[1] + image.shape[0],
    delta_origine_down[0]:delta_origine_down[0] + image.shape[1]] = image

    return arr


def open_scene(slide, origine, downscale, size, scene):
    scale = 1 / downscale
    bbox = slide.get_mosaic_scene_bounding_box(scene)
    max_size_fit = bbox.w, bbox.h
    size_down = int(size * downscale)
    size_fit = (
        max(min(max_size_fit[0], origine[0] + size_down) - origine[0], 0),
        max(min(max_size_fit[1], origine[1] + size_down) - origine[1], 0),
    )

    delta_origine = max(0, -origine[0]), max(0, -origine[1])
    delta_origine_down = ceil(delta_origine[0] / downscale), ceil(delta_origine[1] / downscale)

    arr = np.zeros((size, size, 3))
    try:
        image = slide.read_mosaic((origine[0] + bbox.x + delta_origine[0], origine[1] + bbox.y + delta_origine[1],
                                   size_fit[0] - delta_origine[0], size_fit[1] - delta_origine[1]),
                                  C=0, scale_factor=scale)
    except Exception:
        print("failure for", slide, origine, downscale, size, max_size_fit)
        raise
    image = image.reshape(image.shape[-3:])

    # here it's reversed
    arr[delta_origine_down[1]:delta_origine_down[1] + image.shape[0],
    delta_origine_down[0]:delta_origine_down[0] + image.shape[1]] = image

    return arr


def patch_from_mask(slide, mask, origin, downscale, size, background=0):
    bbox = slide.get_mosaic_bounding_box()
    # here it's reversed
    slide_size = bbox.h, bbox.w
    mask = np.asarray(mask)[:, :, :3]
    mask_size = mask.shape[:2]

    assert np.isclose(mask_size[0] / slide_size[0], mask_size[1] / slide_size[1], rtol=1e-2, atol=1e-2)

    conversion_factor = mask_size[0] / slide_size[0] * downscale

    # reversed
    j, i = origin[0] / downscale, origin[1] / downscale

    shift_x = max(0, -int(i * conversion_factor))
    shift_y = max(0, -int(j * conversion_factor))

    arr = mask[int(i * conversion_factor) + shift_x:int((i + size) * conversion_factor),
               int(j * conversion_factor) + shift_y:int((j + size) * conversion_factor)]

    if arr.dtype == bool:
        arr = arr * 255

    scaled_size = int(size * conversion_factor) + 1
    complete_sub_mask = np.zeros((scaled_size, scaled_size, arr.shape[2]), dtype=np.uint8)
    complete_sub_mask.fill(background)
    complete_sub_mask[shift_x:shift_x + arr.shape[0], shift_y:shift_y + arr.shape[1]] = arr

    final_mask = resize(complete_sub_mask, (size, size))

    res = np.round(final_mask).astype(int).astype(np.uint8)
    return res


class SlideMaskHandler(DataHandler):
    def __init__(self, slide_paths, masks_root, mask_name):
        self.name = "slide_mask"
        self.slide_paths = slide_paths
        self.map_path_name = {basename(x): str(x) for x in self.slide_paths}
        self.masks_root = Path(masks_root)
        self.mask_name = mask_name

        self.cache = dict()

    def get_slide(self, slide_name):
        if slide_name not in self.cache:
            self.cache[slide_name] = aicspylibczi.CziFile(self.map_path_name.get(slide_name))

        return self.cache[slide_name]

    def load_image(self, element):
        assert isinstance(element, dict), "Element is not a dict !"
        assert all([k in element for k in ["slide_name", "downsample", "origin", "size"]])

        slide = self.get_slide(element["slide_name"])

        # handle size conflicts
        return open_image(slide, element["origin"], element["downsample"], element["size"])

    def load_mask(self, element):
        if self.masks_root is None:
            raise AttributeError("Your handler has not been initialized correctly, no mask available")
        assert isinstance(element, dict), "Element is not a dict !"
        assert all([k in element for k in ["slide_name", "downsample", "origin", "size"]])

        slide = self.get_slide(element["slide_name"])

        mask = load_mask_as_image(self.masks_root, element["slide_name"], self.mask_name)

        mask = np.asarray(mask)
        if mask.ndim == 3:
            mask = mask[:, :, :3]
        else:
            mask = mask.reshape(mask.shape + (1,))

        return patch_from_mask(slide, mask,
                               (element["origin"]),
                               element["downsample"], element["size"])


class MultiSlideMaskHandler(SlideMaskHandler):
    def __init__(self, slide_paths, masks_root, mask_name):
        super().__init__(slide_paths, masks_root, mask_name)
        self.name = "multi_slide_mask"

    def load_image(self, element):
        assert isinstance(element, dict), "Element is not a dict !"
        assert all([k in element for k in ["slide_name", "downsample", "origin", "size",
                                           "downscales"]])

        slide = self.get_slide(element["slide_name"])
        images = [open_image(slide, element["origin"], element["downsample"], element["size"])]

        for scale in element["downscales"]:
            lowres_ori = (
                 int(element["origin"][0] + (element["downsample"] - scale) * element["size"] / 2),
                 int(element["origin"][1] + (element["downsample"] - scale) * element["size"] / 2),
            )
            images.append(open_image(slide, lowres_ori,
                          scale, element["size"]))

        return images

    def load_mask(self, element):
        if self.masks_root is None:
            raise AttributeError("Your handler has not been initialized correctly, no mask available")
        assert isinstance(element, dict), "Element is not a dict !"
        assert all([k in element for k in ["slide_name", "downsample", "origin", "size",
                                           "downscales"]])

        slide = self.get_slide(element["slide_name"])

        mask = load_mask_as_image(self.masks_root, element["slide_name"], self.mask_name)

        mask = np.asarray(mask)
        if mask.ndim == 3:
            mask = mask[:, :, :3]
        else:
            mask = mask.reshape(mask.shape + (1,))

        masks = [patch_from_mask(slide, mask, (element["origin"]),
                                 element["downsample"], element["size"])]

        for scale in element["downscales"]:
            lowres_ori = (
                 int(element["origin"][0] + (element["downsample"] - scale) * element["size"] / 2),
                 int(element["origin"][1] + (element["downsample"] - scale) * element["size"] / 2),
            )
            masks.append(patch_from_mask(slide, mask, lowres_ori, scale, element["size"]))

        return masks

