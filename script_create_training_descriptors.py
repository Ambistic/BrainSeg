"""
reate the descriptors to train the segmentation model
"""
import argparse
import os
from pathlib import Path

import aicspylibczi

from brainseg.utils import read_txt, get_slide_size, save_data, build_patches


def load_mask(filename):
    txt = read_txt(filename)
    assert len(txt) >= 2, f"{filename} does not have the 2 lines required"
    mask_geo, mask_slide = txt[0].strip(), txt[1].strip()
    assert "%s" in mask_geo and "%s" in mask_slide, "Missing %s in masks"

    return mask_geo, mask_slide


def export_patches(pairs_geo_slides, name, outdir):
    all_patches = []
    template = dict(
        slidepath=None,
        size=224,
        downscales=[32, 128],
        downscales_masks=[32, 128],
        mask=None,
        structures=["auto_wm", "auto_outline"]
    )

    for geo, str_slide in pairs_geo_slides:
        slide = aicspylibczi.CziFile(str_slide)
        size = get_slide_size(slide)
        current_template = template.copy()
        current_template["slidepath"] = str_slide
        current_template["mask"] = geo
        patches = build_patches(size, 8, 224, template_desc=current_template)
        all_patches += patches
    save_data(all_patches, Path(outdir) / f"{name}.desc")
    print(len(all_patches), "exported for", name)


def fetch_geo_slide_pairs(root: Path, mask_geo, mask_slide):
    pairs = []
    for i in range(500):
        geo = root / "geojson" / (mask_geo % str(i).zfill(3))
        slide = root / "slides" / (mask_slide % str(i).zfill(3))
        if geo.exists() and slide.exists():
            pairs.append((str(geo), str(slide)))

    return pairs


def main(args):
    for dir_ in os.listdir(args.root):
        mask_geo, mask_slide = load_mask(args.root / dir_ / "mask.txt")
        all_geo_slide_pairs = fetch_geo_slide_pairs(args.root / dir_, mask_geo, mask_slide)
        export_patches(all_geo_slide_pairs, dir_, args.outdir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--root", type=Path)
    parser.add_argument("-o", "--outdir", type=Path)

    args_ = parser.parse_args()

    main(args_)
