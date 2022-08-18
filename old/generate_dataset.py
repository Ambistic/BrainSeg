#!/usr/bin/env python
# coding: utf-8
from tqdm import tqdm

from brainseg.slide_provider import get_mask_from_slidepath
from brainseg.streamlit.manager import init_curation_dataset, fill_curation_dataset, is_empty
from brainseg.slide_provider import SlideHandler
from brainseg.provider import provider
from brainseg.loader import Loader

import aicspylibczi
import os
from itertools import product
from pathlib import Path
from PIL import Image
import numpy as np
import re
import argparse
from copy import deepcopy


# In[25]:


def slides_with_mask(slides_path, masks_path):
    keep = []
    for slidepath in os.listdir(slides_path):
        if get_mask_from_slidepath(slidepath, masks_path) is not None:
            keep.append(slidepath)
    return keep


# In[21]:


def parse_curation_name(name):
    try:
        return re.findall("(.+)_\d+_\d+", name)[0]
    except Exception:
        return None


def only_new_slides(slides_path, cur_dts_path):
    current_slides = set(filter(lambda x: x is not None,
                                map(parse_curation_name,
                                    os.listdir(cur_dts_path))))
    slides = list(set(slides_path) - current_slides)
    return slides


def build_czi_patches(fp, downsample, size):
    # non overlapping patches
    slide = aicspylibczi.CziFile(str(fp))
    bbox = slide.get_mosaic_bounding_box()

    patches = [
        dict(
            slidepath=str(fp),
            ori_x=i,
            ori_y=j,
            downscale=downsample,
            size=size
        )
        for i, j in product(
            range(0, bbox.w, size * downsample),
            range(0, bbox.h, size * downsample)
        )
    ]

    return patches


def preprocess_curation_dataset(d, x, y):
    x = x.astype(np.uint8)
    y = y * 255

    data_name = f"{Path(d[1]['slidepath']).name}_{d[1]['ori_x']}_{d[1]['ori_y']}"
    res = dict(
        data_name=data_name,
        image=Image.fromarray(x),
        mask=Image.fromarray(y)
    )
    return res


def filter_patches(args, all_patches):
    if args.filter is None:
        return

    # for each patch
    keep = []
    print("Filtering")
    for patch in tqdm(all_patches):
        # if big mask is empty continue
        modified_patch = deepcopy(patch)
        modified_patch[1]["downscale"] = int(modified_patch[1]["downscale"] / 2)
        modified_patch[1]["ori_x"] = int(modified_patch[1]["ori_x"] - modified_patch[1]["size"] / 2)
        modified_patch[1]["ori_y"] = int(modified_patch[1]["ori_y"] - modified_patch[1]["size"] / 2)

        if is_empty(Image.fromarray(provider.mask(modified_patch) * 255)):
            continue

        # if small mask is not empty continue
        if not is_empty(Image.fromarray(provider.mask(patch) * 255)):
            continue
        # add
        print("Keeping one")
        keep.append(patch)


def main(args):
    keep = slides_with_mask(args.slides, args.masks)
    keep = only_new_slides(keep, args.curated_dataset)
    print(f"Running for {len(keep)} slides")

    all_patches = sum([build_czi_patches(args.slides / sl, args.downsample, args.size) for sl in keep], [])
    all_patches = list(map(lambda x: (args.area, x), all_patches))
    print(f"Running for {len(all_patches)} patches")
    # all_patches = all_patches[:2000]
    print(f"Limiting to {len(all_patches)} patches")

    sh = SlideHandler(args.slides, args.masks, area=args.area)
    provider.register(sh)

    loader = Loader(all_patches, preprocess=preprocess_curation_dataset)

    init_curation_dataset(args.curated_dataset)
    filter_patches(args, all_patches)
    # here we can filter for border
    fill_curation_dataset(args.curated_dataset, loader, name=args.name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--slides", help="Path of the slides")
    parser.add_argument("-m", "--masks", help="Path of the masks")
    parser.add_argument("-c", "--curated_dataset", help="Path of the curated dataset")
    parser.add_argument("-n", "--name", help="Name of the run")
    parser.add_argument("-a", "--area", help="Area to use")
    parser.add_argument("-d", "--downsample", help="Downsample for the images", type=int, default=32)
    parser.add_argument("--size", help="Size of the patches", type=int, default=224)
    parser.add_argument("-f", "--filter", help="Type of filtering", default=None)

    args = parser.parse_args()

    args.slides = Path(args.slides)
    args.masks = Path(args.masks)
    args.curated_dataset = Path(args.curated_dataset)

    print(f"Running with slides : {args.slides}\nmasks: {args.masks}\n"
          f"dataset: {args.curated_dataset}\nname: {args.name}")

    main(args)
    print("Finished !")
