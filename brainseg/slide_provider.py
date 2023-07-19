from copy import deepcopy
from math import ceil

import numpy as np
from pathlib import Path
import aicspylibczi
from skimage.transform import resize
from PIL import Image

from .geo import quickfix_multipolygon_shapely, get_polygon_by_classification
from .path import get_mask_from_slidepath
from .polygon import generate_patch_polygon, rescale_polygon, translate_polygon
from .provider import DataHandler
from .utils import read_histo
from .viz.draw import draw_polygon


def open_image_old(slide, origine, downscale, size):
    scale = 1 / downscale
    bbox = slide.get_mosaic_bounding_box()
    max_size_fit = bbox.w, bbox.h
    arr = np.zeros((size, size, 3))
    size_down = int(size * downscale)
    size_fit = (
        max(min(max_size_fit[0], origine[0] + size_down) - origine[0], 0),
        max(min(max_size_fit[1], origine[1] + size_down) - origine[1], 0),
    )

    image = slide.read_mosaic((origine[0] + bbox.x, origine[1] + bbox.y,
                               size_fit[0], size_fit[1]), C=0, scale_factor=scale)
    image = image.reshape(image.shape[-3:])
    arr[:image.shape[0], :image.shape[1]] = image

    return arr


def open_image(slide, origine, downscale, size, slidename=""):
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
    p = None

    for trial in range(10):
        try:
            image = slide.read_mosaic((origine[0] + bbox.x + delta_origine[0], origine[1] + bbox.y + delta_origine[1],
                                       size_fit[0] - delta_origine[0], size_fit[1] - delta_origine[1]),
                                      C=0, scale_factor=scale)
        except Exception as err:
            p = err
        else:
            p = None
            break

    if p is not None:
        print("failure for", origine, downscale, size, max_size_fit, slidename)
        # raise err
        return arr  # it's not good but will prevent killing everytime

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


def patch_from_mask(slide, mask, origine, downscale, size, background=255):
    bbox = slide.get_mosaic_bounding_box()
    # here it's reversed
    slide_size = bbox.h, bbox.w
    mask = np.asarray(mask)[:, :, :3]

    scale_mask = 12  # shall stay identical for all
    mpp = 0.88
    # from the mask to the downscaled image ?
    conversion_factor_target = downscale / scale_mask * mpp
    conversion_factor_bottom = 1 / scale_mask * mpp

    mid_size = (
        int(mask.shape[0] / 2) - int(slide_size[0] / 2 * conversion_factor_bottom),
        int(mask.shape[1] / 2) - int(slide_size[1] / 2 * conversion_factor_bottom)
    )

    cut_mask = mask[mid_size[0]:-mid_size[0], mid_size[1]:-mid_size[1]]

    scaled_origine = int(origine[0] * conversion_factor_bottom), int(origine[1] * conversion_factor_bottom)
    scaled_size = int(size * conversion_factor_target)

    # here it's reversed
    sub_mask = cut_mask[max(scaled_origine[1], 0):scaled_origine[1] + scaled_size,
                        max(scaled_origine[0], 0):scaled_origine[0] + scaled_size]

    complete_sub_mask = np.zeros((scaled_size, scaled_size, 3), dtype=np.uint8)
    complete_sub_mask.fill(background)
    delta_sub_mask = max(-scaled_origine[1], 0), max(-scaled_origine[0], 0)
    complete_sub_mask[delta_sub_mask[0]:delta_sub_mask[0] + sub_mask.shape[0],
                      delta_sub_mask[1]:delta_sub_mask[1] + sub_mask.shape[1]] = sub_mask

    final_mask = resize(complete_sub_mask, (size, size))

    res = np.round(final_mask).astype(int).astype(np.uint8)
    return res


def patch_from_mask_bck(slide, mask, origine, downscale, size, background=255):
    bbox = slide.get_mosaic_bounding_box()
    # here it's reversed
    slide_size = bbox.h, bbox.w
    mask = np.asarray(mask)[:, :, :3]

    scale_mask = 12  # shall stay identical for all
    mpp = 0.88
    # from the mask to the downscaled image ?
    conversion_factor_target = downscale / scale_mask * mpp
    conversion_factor_bottom = 1 / scale_mask * mpp

    mid_size = (
        int(mask.shape[0] / 2) - int(slide_size[0] / 2 * conversion_factor_bottom),
        int(mask.shape[1] / 2) - int(slide_size[1] / 2 * conversion_factor_bottom)
    )

    cut_mask = mask[mid_size[0]:-mid_size[0], mid_size[1]:-mid_size[1]]

    scaled_origine = int(origine[0] * conversion_factor_bottom), int(origine[1] * conversion_factor_bottom)
    scaled_size = int(size * conversion_factor_target)

    # here it's reversed
    sub_mask = cut_mask[scaled_origine[1]:scaled_origine[1] + scaled_size,
                        scaled_origine[0]:scaled_origine[0] + scaled_size]

    complete_sub_mask = np.zeros((scaled_size, scaled_size, 3), dtype=np.uint8)
    complete_sub_mask.fill(background)
    complete_sub_mask[:sub_mask.shape[0], :sub_mask.shape[1]] = sub_mask

    final_mask = resize(complete_sub_mask, (size, size))

    res = np.round(final_mask).astype(int).astype(np.uint8)
    return res


class SlideHandler(DataHandler):
    def __init__(self, slides_root, masks_root=None, area="whitematter"):
        self.slides_root = Path(slides_root)
        if masks_root is None:
            self.masks_root = None
        else:
            self.masks_root = Path(masks_root)
        self.name = area
        self.area = area

        self.cache = dict()

    def get_slide(self, slidepath):
        fullpath = self.slides_root / slidepath
        if slidepath not in self.cache:
            self.cache[slidepath] = aicspylibczi.CziFile(fullpath)

        return self.cache[slidepath]

    def load_image(self, element):
        assert isinstance(element, dict), "Element is not a dict !"
        assert all([k in element for k in ["slidepath", "downscale", "ori_x", "ori_y", "size"]])

        slide = self.get_slide(element["slidepath"])

        # handle size conflicts
        return open_image(slide, (element["ori_x"], element["ori_y"]), element["downscale"], element["size"])

    def load_mask(self, element):
        if self.masks_root is None:
            raise AttributeError("Your handler has not been initialized correctly, no mask available")
        assert isinstance(element, dict), "Element is not a dict !"
        assert all([k in element for k in ["slidepath", "downscale", "ori_x", "ori_y", "size"]])

        slide = self.get_slide(element["slidepath"])

        maskpath = get_mask_from_slidepath(element["slidepath"], self.masks_root, area=self.area)
        mask = Image.open(maskpath)

        mask = np.asarray(mask)
        if mask.ndim == 3:
            mask = mask[:, :, :3]
        else:
            mask = mask.reshape(mask.shape + (1,))

        return patch_from_mask(slide, mask,
                               (element["ori_x"], element["ori_y"]),
                               element["downscale"], element["size"])


class BiResSlideHandler(SlideHandler):
    def __init__(self, slides_root, masks_root=None, area="whitematter"):
        super().__init__(slides_root, masks_root, area)
        self.name = "bires_" + area

    def load_image(self, element):
        assert isinstance(element, dict), "Element is not a dict !"
        assert all([k in element for k in ["slidepath", "downscale", "ori_x", "ori_y", "size",
                                           "downscale_lowres"]])

        slide = self.get_slide(element["slidepath"])
        lowres_ori = (
             int(element["ori_x"] + (element["downscale"] - element["downscale_lowres"]) * element["size"] / 2),
             int(element["ori_y"] + (element["downscale"] - element["downscale_lowres"]) * element["size"] / 2),
        )

        # handle size conflicts
        highres = open_image(slide, (element["ori_x"], element["ori_y"]), element["downscale"], element["size"])
        lowres = open_image(slide, lowres_ori, element["downscale_lowres"], element["size"])

        return highres, lowres


class MultiMaskSlideHandler(SlideHandler):
    def __init__(self, slides_root, masks_root=None, areas=None, name="default"):
        super().__init__(slides_root, masks_root, "")
        self.name = "multimask_" + name
        self.areas = areas

    def open_mask(self, slidepath, area):
        maskpath = get_mask_from_slidepath(slidepath, self.masks_root, area=area)
        mask = Image.open(maskpath)

        mask = np.asarray(mask)
        if mask.ndim == 3:
            mask = mask[:, :, 0:1]
        else:
            mask = mask.reshape(mask.shape + (1,))

        return ~ mask.astype(bool)

    def load_mask(self, element):
        if self.masks_root is None:
            raise AttributeError("Your handler has not been initialized correctly, no mask available")
        assert isinstance(element, dict), "Element is not a dict !"
        assert all([k in element for k in ["slidepath", "downscale", "ori_x", "ori_y", "size"]])

        slide = self.get_slide(element["slidepath"])

        full_mask = np.zeros((element["size"], element["size"], len(self.areas)), dtype=bool)

        for i, area in enumerate(self.areas):
            mask = self.open_mask(element["slidepath"], area) * 255
            c_mask = patch_from_mask(slide, mask,
                                     (element["ori_x"], element["ori_y"]),
                                     element["downscale"], element["size"],
                                     background=0)
            full_mask[:, :, i] = c_mask[:, :, 0]

        return full_mask


class BiResMultiMaskSlideHandler(BiResSlideHandler, MultiMaskSlideHandler):
    def __init__(self, slides_root, masks_root=None, areas=None, name="default"):
        BiResSlideHandler.__init__(self, slides_root, masks_root, "")
        self.name = "bires_multimask_" + name
        self.areas = areas

    def load_mask(self, element):
        return MultiMaskSlideHandler.load_mask(self, element)

    def load_image(self, element):
        return BiResSlideHandler.load_image(self, element)


class SlideSceneHandler(DataHandler):
    def __init__(self, slides_root, masks_root=None, area="whitematter"):
        self.slides_root = Path(slides_root)
        if masks_root is None:
            self.masks_root = None
        else:
            self.masks_root = Path(masks_root)
        self.name = area
        self.area = area

        self.cache = dict()

    def get_slide(self, slidepath):
        fullpath = self.slides_root / slidepath
        if slidepath not in self.cache:
            self.cache[slidepath] = aicspylibczi.CziFile(fullpath)

        return self.cache[slidepath]

    def load_image(self, element):
        assert isinstance(element, dict), "Element is not a dict !"
        assert all([k in element for k in ["slidepath", "downscale", "ori_x", "ori_y", "size", "scene"]])

        slide = self.get_slide(element["slidepath"])

        # handle size conflicts
        return open_scene(slide, (element["ori_x"], element["ori_y"]),
                          element["downscale"], element["size"], element["scene"])

    def load_mask(self, element):
        if self.masks_root is None:
            raise AttributeError("Your handler has not been initialized correctly, no mask available")
        assert isinstance(element, dict), "Element is not a dict !"
        assert all([k in element for k in ["slidepath", "downscale", "ori_x", "ori_y", "size"]])

        slide = self.get_slide(element["slidepath"])

        maskpath = get_mask_from_slidepath(element["slidepath"], self.masks_root, area=self.area)
        mask = Image.open(maskpath)

        mask = np.asarray(mask)
        if mask.ndim == 3:
            mask = mask[:, :, :3]
        else:
            mask = mask.reshape(mask.shape + (1,))

        return patch_from_mask(slide, mask,
                               (element["ori_x"], element["ori_y"]),
                               element["downscale"], element["size"])


class MultiResSlideHandler(SlideHandler):
    def __init__(self, slides_root, masks_root=None, area="whitematter"):
        super().__init__(slides_root, masks_root, area)
        self.name = "multires_" + area

    def load_image(self, element):
        assert isinstance(element, dict), "Element is not a dict !"
        assert all([k in element for k in ["slidepath", "downscale", "ori_x", "ori_y", "size",
                                           "downscales"]])

        slide = self.get_slide(element["slidepath"])
        images = [open_image(slide, (element["ori_x"], element["ori_y"]),
                             element["downscale"], element["size"])]

        for scale in element["downscales"]:
            lowres_ori = (
                 int(element["ori_x"] + (element["downscale"] - scale) * element["size"] / 2),
                 int(element["ori_y"] + (element["downscale"] - scale) * element["size"] / 2),
            )
            images.append(open_image(slide, lowres_ori,
                          scale, element["size"]))

        return images


def square_from_descriptor(desc):
    return generate_patch_polygon((desc["ori_x"], desc["ori_y"]), desc["size"], desc["downscale"])


class QuPathMultiSlideHandler(DataHandler):
    def __init__(self):
        self.name = "qupath_multi"
        self.cache_slide = dict()
        self.cache_geo = dict()

    def get_slide(self, slidepath):
        if slidepath not in self.cache_slide:
            self.cache_slide[slidepath] = aicspylibczi.CziFile(slidepath)

        return self.cache_slide[slidepath]

    def get_geo(self, slidepath):
        if slidepath not in self.cache_geo:
            self.cache_geo[slidepath] = quickfix_multipolygon_shapely(read_histo(slidepath))

        return self.cache_geo[slidepath]

    def load_image(self, element):

        slide = self.get_slide(element["slidepath"])
        images = [open_image(slide, (element["ori_x"], element["ori_y"]),
                             element["downscale"], element["size"], element["slidepath"])]

        for scale in element["downscales"]:
            lowres_ori = (
                 int(element["ori_x"] + (element["downscale"] - scale) * element["size"] / 2),
                 int(element["ori_y"] + (element["downscale"] - scale) * element["size"] / 2),
            )
            images.append(open_image(slide, lowres_ori,
                          scale, element["size"]))

        return images

    def load_single_mask(self, element):
        square = square_from_descriptor(element)

        slide = self.get_slide(element["slidepath"])
        geo = self.get_geo(element["mask"])

        full_mask = np.zeros((element["size"], element["size"], len(element["structures"])), dtype=bool)

        for i, structure in enumerate(element["structures"]):
            poly_structure = get_polygon_by_classification(geo, structure)
            poly_mask = poly_structure.intersection(square)
            poly_mask_ready = rescale_polygon(translate_polygon(poly_mask, -element["ori_x"], -element["ori_y"]),
                                              1 / element["downscale"])

            arr = np.zeros((element["size"], element["size"]), dtype=bool)
            if poly_mask_ready.area > 0:
                c_mask = draw_polygon(arr, poly_mask_ready)
            else:
                c_mask = arr
            full_mask[:, :, i] = c_mask[:, :].T  # maybe the transpose need to occur before ?

        return full_mask

    def load_mask(self, element):

        masks = [self.load_single_mask(element)]

        for scale in element["downscales_masks"]:
            lowres_ori = (
                 int(element["ori_x"] + (element["downscale"] - scale) * element["size"] / 2),
                 int(element["ori_y"] + (element["downscale"] - scale) * element["size"] / 2),
            )
            element = deepcopy(element)
            element["ori_x"] = lowres_ori[0]
            element["ori_y"] = lowres_ori[1]
            element["downscale"] = scale
            masks.append(self.load_single_mask(element))

        return masks


class MultiSlideHandler(DataHandler):
    def __init__(self):
        self.name = "multi"
        self.cache = dict()

    def get_slide(self, slidepath):
        if slidepath not in self.cache:
            self.cache[slidepath] = aicspylibczi.CziFile(slidepath)

        return self.cache[slidepath]

    def load_image(self, element):

        slide = self.get_slide(element["slidepath"])
        images = [open_image(slide, (element["ori_x"], element["ori_y"]),
                             element["downscale"], element["size"], element["slidepath"])]

        for scale in element["downscales"]:
            lowres_ori = (
                 int(element["ori_x"] + (element["downscale"] - scale) * element["size"] / 2),
                 int(element["ori_y"] + (element["downscale"] - scale) * element["size"] / 2),
            )
            images.append(open_image(slide, lowres_ori,
                          scale, element["size"]))

        return images

    def open_mask(self, mask_path):
            mask = Image.open(mask_path)

            mask = np.asarray(mask)
            if mask.ndim == 3:
                mask = mask[:, :, 0:1]
            else:
                mask = mask.reshape(mask.shape + (1,))

            return (~ mask.astype(bool)) * 255

    def load_single_mask(self, element):

        slide = self.get_slide(element["slidepath"])

        full_mask = np.zeros((element["size"], element["size"], len(element["masks"])), dtype=bool)

        for i, mask_path in enumerate(element["masks"]):
            mask = self.open_mask(mask_path)
            c_mask = patch_from_mask(slide, mask,
                                     (element["ori_x"], element["ori_y"]),
                                     element["downscale"], element["size"],
                                     background=0)
            full_mask[:, :, i] = c_mask[:, :, 0]

        return full_mask

    def load_mask(self, element):

        masks = [self.load_single_mask(element)]

        for scale in element["downscales_masks"]:
            lowres_ori = (
                 int(element["ori_x"] + (element["downscale"] - scale) * element["size"] / 2),
                 int(element["ori_y"] + (element["downscale"] - scale) * element["size"] / 2),
            )
            element = deepcopy(element)
            element["ori_x"] = lowres_ori[0]
            element["ori_y"] = lowres_ori[1]
            element["downscale"] = scale
            masks.append(self.load_single_mask(element))

        return masks