from .streamlit.manager import load_image, load_mask

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
