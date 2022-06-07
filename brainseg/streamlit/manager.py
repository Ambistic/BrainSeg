import datetime
import glob
import shutil
import os
from pathlib import Path

from PIL.Image import Image
from tqdm import tqdm

from .utils import safe_move, parse_mask_val


def init_curation_dataset(path):
    path = Path(path)
    path.mkdir(exist_ok=True, parents=True)

    (path / ".curation").touch()
    (path / ".removed").mkdir(parents=True, exist_ok=True)


def check_valid_path(path):
    assert (Path(path) / ".curation").exists(), \
        "You must first initialize a curation dataset !"

    assert (Path(path) / ".removed").exists(), \
        "Something went wrong, .removed folder is missing !"


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
    return source, target


def reset(last_source, last_target):
    shutil.move(last_target, last_source)


def is_empty(image: Image, background="white"):
    reference = (0, 0) if background == "black" else (255, 255)
    extrema = image.convert("L").getextrema()

    return extrema == reference


def fill_curation_dataset(path, data, name=None, keep_empty_masks=False):
    """

    :param keep_empty_masks:
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
    for element in tqdm(data):
        if not keep_empty_masks:
            if is_empty(element["mask"]):
                continue
        add_element(path, **element, mask_name=name)


def get_list_mask(path, data_name):
    alls = glob.glob(str(Path(path) / data_name / "mask" / f"mask_*.png"))
    return [Path(x).name for x in alls]


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
        if not (path / sample).is_dir() or sample.startswith("."):
            continue

        mask_name, val = get_best_mask(path, sample)
        if mask_name is not None and val >= min_threshold:
            res.append(dict(
                path=str(path),
                data_name=sample,
                mask_name=mask_name,
            ))

    return [("image", x) for x in res]


def has_zero(path, sample):
    path = Path(path)
    for mask_name in os.listdir(path / sample / "mask"):
        val = parse_mask_val(mask_name)
        if val == 0:
            return True
    return False


def get_ones(path, sample):
    path = Path(path)
    ls = []
    for mask_name in os.listdir(path / sample / "mask"):
        val = parse_mask_val(mask_name)
        if val == 1:
            ls.append(mask_name)
    return ls


def flush_out(path):
    path = Path(path)
    check_valid_path(path)

    # create the current remove directory
    folder = path / ".removed" / datetime.datetime.now().isoformat().replace(":", "-")
    folder.mkdir()

    # loop over sample
    for sample in os.listdir(path):
        if not (path / sample).is_dir() or sample.startswith("."):
            continue

        ones = get_ones(path, sample)

        # loop over content
        for one in ones:
            safe_move(path / sample / "mask" / one,
                      folder / sample / "mask" / one)
