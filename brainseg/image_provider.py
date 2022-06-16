from .streamlit.load import load_mask, load_image, load_lowres

from .provider import DataHandler
import numpy as np

from .utils import rgb_to_multi


class ImageHandler(DataHandler):
    name = "image"

    def __init__(self):
        pass

    def load_image(self, element):
        assert isinstance(element, dict), "Element is not a dict !"
        assert all([k in element for k in ["path", "data_name", "mask_name"]])

        return load_image(element["path"], element["data_name"])

    def load_mask(self, element):
        assert isinstance(element, dict), "Element is not a dict !"
        assert all([k in element for k in ["path", "data_name", "mask_name"]])

        return load_mask(element["path"], element["data_name"], element["mask_name"])


class BiResImageHandler(ImageHandler):
    name = "bires_image"

    def load_image(self, element):
        assert isinstance(element, dict), "Element is not a dict !"
        assert all([k in element for k in ["path", "data_name", "mask_name"]])

        return [
            load_image(element["path"], element["data_name"]),
            load_lowres(element["path"], element["data_name"])
        ]


class MultiMaskImageHandler(ImageHandler):
    name = "image"

    def __init__(self, colors):
        super().__init__()
        self.colors = colors

    def load_mask(self, element):
        assert isinstance(element, dict), "Element is not a dict !"
        assert all([k in element for k in ["path", "data_name", "mask_name"]])

        mask = load_mask(element["path"], element["data_name"], element["mask_name"])
        mask = np.asarray(mask)
        mask = rgb_to_multi(mask, self.colors)
        return mask


class BiResMultiMaskImageHandler(MultiMaskImageHandler, BiResImageHandler):
    name = "image"

    def __init__(self, colors):
        MultiMaskImageHandler.__init__(self, colors)

    def load_mask(self, element):
        return MultiMaskImageHandler.load_mask(self, element)

    def load_image(self, element):
        return BiResImageHandler.load_image(self, element)
