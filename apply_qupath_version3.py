"""
This script is an improvement of the version 2, it processes images in batch.
It stills requires 3-resolution models

This scripts still requires to be tested !!!
"""
import argparse
import math
import os.path
import traceback
from pathlib import Path

import aicspylibczi
import numpy as np
from PIL import Image
from scipy import ndimage
from tqdm import tqdm

from brainseg.config import fill_with_config
from brainseg.models.multires_unet5 import multires_unet
from brainseg.models.utils import transfer_weights
from brainseg.path import build_path_histo, build_path_histo_segmentation
from brainseg.provider import provider
import segmentation_models as sm
import datetime

from brainseg.slide_provider import MultiSlideHandler

sm.set_framework("tf.keras")

MODEL = None
N_MASK = 2


def _get_model(args):
    model_full = multires_unet(n_res=3, n_classes=2, im_sz=224, n_channels=3,
                          n_filters_start=32, growth_factor=1.2, upconv=True,
                          all_outputs=True)

    model_red = multires_unet(n_res=3, n_classes=2, im_sz=224, n_channels=3,
                               n_filters_start=32, growth_factor=1.2, upconv=True,
                               all_outputs=False)

    model_full.load_weights(args.segmentation_weights)
    transfer_weights(model_full, model_red)

    return model_red


def get_model(args):
    global MODEL

    if MODEL is None:
        # ONLY THIS LINE HAS TO BE CHANGED
        model = _get_model(args)
        model.compile()
        MODEL = model

    return MODEL


def post_processing(mask):
    if mask.ndim == 2:
        mask = mask.reshape((*mask.shape, 1))

    mask = ndimage.gaussian_filter(mask, sigma=(5, 5, 0), order=0)
    mask = (mask * 255.).astype(int).astype(np.uint8).reshape(mask.shape[:2])

    return mask


def segment(args, slide_path, x, y, size):
    model = get_model(args)
    desc = dict(
        slidepath=slide_path,
        size=size,
        downscales=[32, 128],
        ori_x=x,
        ori_y=y,
        downscale=args.downscale,
    )
    image = provider.image(("multi", desc))
    image = list(map(lambda im: np.array([np.asarray(im) / 255.]), image))

    # the second [0] is because the model has 2 outputs
    mask = model.predict(image, verbose=False)[0]
    mask = np.rollaxis(mask, 2)

    mask = list(map(post_processing, mask))

    return mask


def segment_batch(args, slide_path, batch_x, batch_y, size):
    model = get_model(args)
    descs = [dict(
        slidepath=slide_path,
        size=size,
        downscales=[32, 128],
        ori_x=x * args.downscale,
        ori_y=y * args.downscale,
        downscale=args.downscale,
    ) for x, y in zip(batch_x, batch_y)]

    images = [provider.image(("multi", desc)) for desc in descs]
    images = list(map(lambda im: np.array(np.asarray(list(im)) / 255.), zip(*images)))

    masks = model.predict(images, verbose=False)
    masks = [np.rollaxis(mask, 2) for mask in masks]

    masks = [list(map(post_processing, mask)) for mask in masks]

    return masks


def generate_coords(size, margin, full_size):
    patch_size = size - 2 * margin
    limit_size = full_size[0] + margin - size, full_size[1] + margin - size

    for i in range(-margin, limit_size[0], patch_size):
        for j in range(-margin, limit_size[1], patch_size):
            yield i, j


def add_patch(full_mask, mask, x, y, size, margin):
    # here it's reversed
    patch = mask[margin:-margin, margin:-margin].transpose()

    # top left is fine but bottom right can oversize
    true_size_x = min(full_mask.shape[0], x + size - margin) - x
    true_size_y = min(full_mask.shape[1], y + size - margin) - y

    full_mask[x + margin:x + true_size_x,
              y + margin:y + true_size_y] = patch


def init_provider(args):
    sh = MultiSlideHandler()
    provider.register(sh)


def get_slide_size(slide_path, downscale=1):
    """Reverse the axes"""
    slide = aicspylibczi.CziFile(slide_path)
    bbox = slide.get_mosaic_bounding_box()
    return math.ceil(bbox.w / downscale), math.ceil(bbox.h / downscale)


def save_mask(full_mask, save_path):
    # img = Image.fromarray(full_mask * 255)
    img = Image.fromarray(full_mask)
    img.save(save_path)


def batch_generator(iterator, n):
    batch = []
    for item in iterator:
        batch.append(item)
        if len(batch) == n:
            yield batch
            batch = []
    if len(batch) > 0:
        print(len(batch))
        yield batch


def segment_slide(args, slide_path, slice_id):
    print("Initializing ...")
    full_size = get_slide_size(slide_path, args.downscale)

    full_mask = [np.zeros(full_size, dtype=np.uint8) for _ in range(N_MASK)]

    print("Running segmentation")
    for batch in tqdm(list(batch_generator(
            generate_coords(args.size, args.margin, full_size), n=16)
            )):
        batch_x, batch_y = zip(*batch)
        batch_mask = segment_batch(args, slide_path, batch_x, batch_y, args.size)
        for mask, x, y in zip(batch_mask, batch_x, batch_y):
            [add_patch(full_mask[i], mask[i], x, y, args.size, args.margin) for i in range(N_MASK)]

    print("Saving mask")
    mask_types = ["gm", "pial"]
    for i in range(N_MASK):
        save_mask(full_mask[i].transpose(), build_path_histo_segmentation(
            args.annotations_dir, slice_id, args.annotations_mask, mask_types[i], "mask"
        ))


def main(args):
    init_provider(args)

    for slice_id in range(args.start, args.end, args.step):
        slide_path = build_path_histo(args.slides_dir, slice_id, args.slides_mask)
        if not os.path.exists(slide_path):
            print(slide_path, "does not exist, skipping")
            continue
        try:
            segment_slide(args, slide_path, slice_id)
        except Exception as e:
            print("Couldn't segment slide", slice_id, "reason:", e)
            traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=Path, default=Path("/media/tower/LaCie/REGISTRATION_PROJECT/"
                                                                  "TMP_PIPELINE_M148/config.ini"))
    parser.add_argument("--slides_dir", type=Path, default=None)
    parser.add_argument("--slides_mask", type=str, default=None)
    parser.add_argument("--segmentation_weights", type=str, default=None)
    parser.add_argument("--annotations_dir", type=str, default=None)
    parser.add_argument("--annotations_mask", type=str, default=None)
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--margin", type=int, default=16)  # old 56
    parser.add_argument("--size", type=int, default=224)  # old 224
    parser.add_argument("-d", "--downscale", type=int, default=8)

    args_ = fill_with_config(parser)

    main(args_)
