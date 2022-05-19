import glob
import numpy as np
import shutil
import os
from pathlib import Path
from PIL import Image
import re


def init_curation_dataset(path):
    path = Path(path)
    path.mkdir(exist_ok=True, parents=True)

    (path / ".curation").touch()


def check_valid_path(path):
    assert (Path(path) / ".curation").exists(), \
        "You must first initialize a curation dataset !"


def add_element(path: Path, data_name=None, image=None, mask_name=None, mask=None):
    check_valid_path(path)

    if not (path / data_name).exists():
        if image is None:
            raise ValueError("Cannot create a sample without an image")
        (path / data_name).mkdir()

    if image is not None:
        if (path / data_name / "image.png").exists():
            raise ValueError("An image already exist")
        image.save(str(path / data_name / "image.png"))

    if mask is not None:
        (path / data_name / "mask").mkdir(exist_ok=True)
        assert mask_name is not None, "Cannot have a mask without a mask_name"
        assert len(glob.glob(str(path / data_name / "mask" / f"mask_{mask_name}_*.png"))) == 0, \
            f"A mask already exist for {data_name} and {mask_name}"

        mask.save(str(path / data_name / "mask" / f"mask_{mask_name}_000.png"))


def change_priority(path: Path, data_name, mask_name, priority: int):
    path = Path(path)
    check_valid_path(path)
    assert (priority >= 0) and (priority < 1000), "Priority is not correct, shall be between 0 and 1000"

    source = glob.glob(str(path / data_name / "mask" / f"mask_{mask_name}_*.png"))[0]
    target = Path(source).parent / f"mask_{mask_name}_{str(priority).zfill(3)}.png"

    shutil.move(source, target)


def parse_mask_name(fn):
    return re.findall(r"mask_([\w\d]+)_\d+.png", fn)[0]


def parse_mask_val(fn):
    return int(re.findall(r"mask_[\w\d]+_(\d+).png", fn)[0])


def fill_curation_dataset(path, data, name=None):
    """

    :param path: path of the curation dataset
    :param data: list of dict having data_name key at least
    Other possible keys are : image and mask
    :param name: name of the current set of masks
    :return:
    """
    path = Path(path)
    check_valid_path(path)
    # shall take the reference as argument then call the provider
    # or a 2/3-uple (name, image, mask) or (name, mask)
    # with an error if the image does not exist
    for element in data:
        add_element(path, **element, mask_name=name)


def get_list_mask(path, data_name):
    alls = glob.glob(str(Path(path) / data_name / "mask" / f"mask_*.png"))
    return [Path(x).name for x in alls]


def load_mask(path, data_name, mask_name):
    fp = Path(path) / data_name / "mask" / mask_name
    image = Image.open(fp)
    return image


def load_image(path, data_name):
    fp = Path(path) / data_name / "image.png"
    image = Image.open(fp)
    return image


# noinspection PyTypeChecker
def load_superpose_mask(path, data_name, mask_name):
    mask = np.asarray(load_mask(path, data_name, mask_name))
    image = np.asarray(load_image(path, data_name))

    new_img = np.maximum(~mask.astype(bool), 0.5) * image
    return new_img.astype(int).astype(np.uint8)


# noinspection PyTypeChecker
def load_multiply_mask(path, data_name, mask_name):
    mask = np.asarray(load_mask(path, data_name, mask_name))
    image = np.asarray(load_image(path, data_name))

    return ~mask.astype(bool) * image


def get_best_mask(path, sample):
    path = Path(path)
    best, best_val = None, -1
    for mask_name in os.listdir(path / sample / "mask"):
        val = parse_mask_val(mask_name)
        if val > best_val:
            best, best_val = mask_name, val

    return best, best_val


def list_all(path, min_threshold):
    path = Path(path)
    res = []

    for sample in os.listdir(path):
        if not (path / sample).is_dir():
            continue

        mask_name, val = get_best_mask(path, sample)
        if mask_name is not None and val >= min_threshold:
            res.append(dict(
                path=str(path),
                data_name=sample,
                mask_name=mask_name,
            ))

    return [("image", x) for x in res]
