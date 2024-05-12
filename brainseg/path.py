import os
from pathlib import Path


def get_mask_from_slidepath(slidepath, masks_root, area="whitematter"):
    stem = Path(slidepath).stem
    fp = Path(masks_root) / (stem + 'seg') / (area + ".png")
    if fp.exists():
        return fp
    raise FileExistsError("Does not exist !")


def build_path_mri(rootdir, section_id, type_mri):
    assert type_mri in ["gm", "raw", "pial", "wm"]
    section_id = str(section_id).zfill(3)

    return f"{rootdir}/{type_mri}_{section_id}.png"


def build_path_histo(rootdir, section_id, filename_mask):
    assert "%s" in filename_mask
    section_id = str(section_id).zfill(3)

    return os.path.join(rootdir, filename_mask % section_id)


def build_path_from_mask(rootdir, section_id, filename_mask, zfill=False):
    assert "%s" in filename_mask
    section_id = str(section_id)
    if zfill:
        section_id = section_id.zfill(3)

    return os.path.join(rootdir, filename_mask % section_id)


def build_path_histo_segmentation(rootdir, section_id, filename_mask, type_segmentation, format_segmentation):
    assert "%s" in filename_mask
    section_id = str(section_id).zfill(3)
    format_to_extension = dict(mask="png", geojson="geojson", svg="svg")
    extension = format_to_extension.get(format_segmentation)
    if extension is None:
        raise ValueError(f"Argument `format_segmentation` not recognized. {format_segmentation} not among "
                         f"this list : {list(format_to_extension.values())}")

    name = (filename_mask % section_id) + f"_{type_segmentation}.{extension}"

    return os.path.join(rootdir, name)
