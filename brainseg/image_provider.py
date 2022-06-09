from .streamlit.load import load_mask, load_image, load_lowres

from .provider import DataHandler, provider


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
