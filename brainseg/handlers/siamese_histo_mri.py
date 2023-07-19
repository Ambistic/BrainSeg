from copy import deepcopy
from os.path import basename
from random import choice

from brainseg.provider import DataHandler
import numpy as np
from nilearn import image
import nibabel as nib
import re
from PIL import Image

_CACHE = dict()
_IMAGE_CACHE = dict()


def affine_matrix(angles=None, trans=None):
    """
    :param angles: dict with keys x, y and z
    :param trans: dict with keys x, y and z
    """
    deg_to_rad = np.pi / 180
    if angles is None:
        angles = dict()
    if trans is None:
        trans = dict()
    cos_theta = np.cos(angles.get("z", 0) * deg_to_rad)
    sin_theta = np.sin(angles.get("z", 0) * deg_to_rad)
    rot_mat_z = np.array([[cos_theta, -sin_theta, 0, 0],
                          [sin_theta, cos_theta, 0, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]])

    cos_psi = np.cos(angles.get("y", 0) * deg_to_rad)
    sin_psi = np.sin(angles.get("y", 0) * deg_to_rad)
    rot_mat_y = np.array([[cos_psi, 0, sin_psi, 0],
                          [0, 1, 0, 0],
                          [-sin_psi, 0, cos_psi, 0],
                          [0, 0, 0, 1]])

    cos_gamma = np.cos(angles.get("x", 0) * deg_to_rad)
    sin_gamma = np.sin(angles.get("x", 0) * deg_to_rad)
    rot_mat_x = np.array([[1, 0, 0, 0],
                          [0, cos_gamma, -sin_gamma, 0],
                          [0, sin_gamma, cos_gamma, 0],
                          [0, 0, 0, 1]])

    trans_mat = np.array([
        [1, 0, 0, trans.get("x", 0)],
        [0, 1, 0, trans.get("y", 0)],
        [0, 0, 1, trans.get("z", 0)],
        [0, 0, 0, 1],
    ])

    return rot_mat_y @ rot_mat_z @ rot_mat_x @ trans_mat


def affine_image(img, affine):
    after_rot = image.resample_img(img,
                                   target_affine=affine @ img.affine,
                                   target_shape=img.shape[:3],
                                   interpolation='continuous')
    return after_rot


def extract_monkey_name(name):
    return re.findall(r".*([MC]\d{3}).*(\d{3}).*", name)[0][0]


def extract_slide_id(name):
    return re.findall(r".*([MC]\d{3}).*(\d{3}).*", name)[0][1]


def extract_monkey_from_mri(name):
    return re.findall(r".*([MC]\d{3}).*", name)[0]


def oasis_path_to_mask(path):
    return path.replace("t88_gfc", "t88_masked_gfc")


def load_mri(name, rotx, rotz):
    if not (name, rotx, rotz) in _CACHE:
        img = nib.load(name)
        if rotx != 0 or rotz != 0:
            img = affine_image(img, affine_matrix(dict(x=rotx, y=0, z=rotz)))
        _CACHE[(name, rotx, rotz)] = img

    return _CACHE[(name, rotx, rotz)]


def load_histo_image(desc):
    if desc["image_name"] not in _IMAGE_CACHE:
        _IMAGE_CACHE[desc["image_name"]] = Image.open(desc["image_name"])

    return _IMAGE_CACHE[desc["image_name"]]


def load_mri_image(desc):
    desc = deepcopy(desc)
    if desc.get("has_mask", False):
        if choice([True, False]):
            desc["mri_name"] = oasis_path_to_mask(desc["mri_name"])

    mri_image = load_mri(desc["mri_name"], desc["rotx"], desc["rotz"])
    img = mri_image.get_fdata()[:, desc["mri_id"], :]
    return img


def load_image(desc):
    if desc["type"] == "mri":
        return load_mri_image(desc)

    elif desc["type"] == "histo":
        return load_histo_image(desc)

    else:
        raise AttributeError("Unrecognised descriptor, type provided is unknown")


def process_mapping(mapping, monkey_name, slide_id):
    df = mapping.set_index(["monkey_name", "histo_id"])
    row = df.loc[(monkey_name, slide_id)]
    return row["rot_x"], row["rot_z"], row["mri_id"]


def list_descriptors(slides, dict_mris, mapping):
    descriptors = []
    for image_name in slides:
        name = basename(image_name)
        monkey_name = extract_monkey_name(name)

        slide_id = int(extract_slide_id(name))
        try:
            rotx, rotz, mri_id = process_mapping(mapping, monkey_name, slide_id)
        except:
            # mapping failed, no mri available
            continue
        desc = dict(
            image_name=image_name,
            mri_name=dict_mris[monkey_name],
            monkey_name=monkey_name,
            slide_id=slide_id,
            rotx=rotx,
            rotz=rotz,
            mri_id=mri_id,
        )
        descriptors.append(desc)
    return descriptors


def list_descriptors_histo_only(slides):
    descriptors = []
    for image_name in slides:
        name = basename(image_name)
        monkey_name = extract_monkey_name(name)

        slide_id = int(extract_slide_id(name))
        desc = dict(
            image_name=image_name,
            monkey_name=monkey_name,
            slide_id=slide_id,
        )
        descriptors.append(desc)
    return descriptors


def list_descriptors_mri_only(images, extractor=extract_monkey_name, has_mask=True):
    # for each mri file
    # cut -20 and +20, and generate a "slide" per index
    descriptors = []
    for image_name in images:
        nib_img = nib.load(image_name)
        size = nib_img.get_fdata().shape[1]

        for index in range(+20, size - 20):
            desc = dict(
                mri_name=image_name,
                monkey_name=extractor(image_name),
                slide_id=index,  # for triplet construction
                rotx=0,
                rotz=0,
                mri_id=index,
                has_mask=has_mask,  # for mask switch
            )
            descriptors.append(desc)

    return descriptors


class SiameseHistoMRIHandler(DataHandler):
    def __init__(self):
        super().__init__()
        self.name = "siamese_histo_mri"

    def load_image(self, element):
        """
        """
        assert isinstance(element, dict), "Element is not a dict !"
        assert all([k in element for k in ["image_name", "mri_name", "positive_rotx",
                                           "positive_rotz", "positive_mri_id", "negative_rotx",
                                           "negative_rotz", "negative_mri_id"]])

        origin_image = load_histo_image(element)
        positive_image = load_mri_image(dict(mri_name=element["mri_name"],
                                             rotx=element["positive_rotx"],
                                             rotz=element["positive_rotz"],
                                             mri_id=element["positive_mri_id"]))
        negative_image = load_mri_image(dict(mri_name=element["mri_name"],
                                             rotx=element["negative_rotx"],
                                             rotz=element["negative_rotz"],
                                             mri_id=element["negative_mri_id"]))

        return [origin_image, positive_image, negative_image]

    def load_mask(self, element):
        return None


class SiameseHistoMRIHandler2(DataHandler):
    def __init__(self):
        super().__init__()
        self.name = "siamese_histo_mri"

    def load_image(self, element):
        """
        """
        assert isinstance(element, dict), "Element is not a dict !"
        assert all([k in element for k in ["anchor", "positive", "negative"]])

        anchor_image = load_image(element["anchor"])
        positive_image = load_image(element["positive"])
        negative_image = load_image(element["negative"])

        return [anchor_image, positive_image, negative_image]

    def load_mask(self, element):
        return None


class SelfHistoMRIHandler(DataHandler):
    def __init__(self):
        super().__init__()
        self.name = "self_histo_mri"

    def load_image(self, element):
        """
        """
        assert isinstance(element, dict), "Element is not a dict !"

        return load_image(element)

    def load_mask(self, element):
        return self.load_image(element)


class SingleImageHandler(DataHandler):
    def __init__(self):
        super().__init__()
        self.name = "single_image"

    def load_image(self, element):
        """
        """
        assert isinstance(element, dict), "Element is not a dict !"

        return load_image(element)

    def load_mask(self, element):
        return None
