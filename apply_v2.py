import argparse
import math
from pathlib import Path

import aicspylibczi
import numpy as np
from PIL import Image
from scipy import ndimage
from tqdm import tqdm

from brainseg.bires_unet import bi_res_unet
from brainseg.provider import provider
import segmentation_models as sm

from brainseg.slide_provider import SlideHandler, BiResSlideHandler

sm.set_framework("tf.keras")

MODEL = None


def get_model(args):
    global MODEL

    if MODEL is None:
        model = bi_res_unet(n_classes=1, im_sz=224, n_channels=3, n_filters_start=32,
                            growth_factor=1.2, upconv=True)
        model.load_weights(args.weights)
        model.compile()
        MODEL = model

    return MODEL


def segment(args, x, y, size):
    model = get_model(args)
    image = provider.image(("bires_slide", dict(
        slidepath=args.slide_path,
        downscale=args.downscale,
        downscale_lowres=64,
        ori_x=x,
        ori_y=y,
        size=size,
    )))
    img_high, img_low = image
    img_high = np.asarray(img_high) / 255.
    img_low = np.asarray(img_low) / 255.
    mask = model.predict([np.array([img_high]), np.array([img_low])])[0]
    # mask = (mask > 0.5).astype(int).astype(np.uint8).reshape(mask.shape[:2])
    mask = ndimage.gaussian_filter(mask, sigma=(5, 5, 0), order=0)
    mask = (mask * 255.).astype(int).astype(np.uint8).reshape(mask.shape[:2])

    return mask


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
    sh = BiResSlideHandler(args.slide_path.parent, area="slide")
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


def main(args):
    print("Initializing ...")
    full_size = get_slide_size(args.slide_path, args.downscale)
    full_mask = np.zeros(full_size, dtype=np.uint8)
    init_provider(args)
    print("Running segmentation")
    for x, y in tqdm(list(generate_coords(args.size, args.margin, full_size))):
        mask = segment(args, x * args.downscale, y * args.downscale, args.size)
        add_patch(full_mask, mask, x, y, args.size, args.margin)

    print("Saving mask")
    save_mask(full_mask.transpose(), args.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--slide-path")
    parser.add_argument("-d", "--downscale", type=int, default=32)
    parser.add_argument("-w", "--weights", default="model_iou_0.9604.h5")
    parser.add_argument("--save-path", default="test.png")
    parser.add_argument("--margin", type=int, default=56)
    parser.add_argument("--size", type=int, default=224)

    args = parser.parse_args()
    args.slide_path = Path(args.slide_path)

    main(args)
