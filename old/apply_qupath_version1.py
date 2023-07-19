# THIS VERSION IS FOR FULL TISSUE SEGMENTATION WITH ALL STAININGS (NEUN, CV, ETC)
# IT USES A 2-SCALE MODEL TO PREDICT AT A DOWNSAMPLE OF 8
import argparse
import math
from pathlib import Path

import aicspylibczi
import numpy as np
from PIL import Image
from scipy import ndimage
from tqdm import tqdm

from brainseg.models.multires_unet4 import multires_unet
from brainseg.provider import provider
import segmentation_models as sm
import datetime

from brainseg.slide_provider import MultiSlideHandler

sm.set_framework("tf.keras")

MODEL = None
N_MASK = 2


def get_model(args):
    global MODEL

    if MODEL is None:
        # ONLY THIS LINE HAS TO BE CHANGED
        model = multires_unet(n_res=3, n_classes=2, im_sz=224, n_channels=3,
                              n_filters_start=32, growth_factor=1.2, upconv=True)
        model.load_weights(args.weights)
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


def segment_slide(args, slide_path):
    print("Initializing ...")
    full_size = get_slide_size(slide_path, args.downscale)

    full_mask = [np.zeros(full_size, dtype=np.uint8) for _ in range(N_MASK)]

    print("Running segmentation")
    for x, y in tqdm(list(generate_coords(args.size, args.margin, full_size))):
        mask = segment(args, slide_path, x * args.downscale, y * args.downscale, args.size)
        [add_patch(full_mask[i], mask[i], x, y, args.size, args.margin) for i in range(N_MASK)]

    print("Saving mask")
    for i in range(N_MASK):
        save_mask(full_mask[i].transpose(), args.save_dir / (slide_path.stem + f"_{i}.png"))


def main(args):
    length = len(args.slide_paths)
    init_provider(args)
    for i, slide_path in enumerate(args.slide_paths):
        print(f"Starting segmentation for {i} / {length} : {slide_path}")
        segment_slide(args, slide_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--slide-paths", nargs="+")
    parser.add_argument("-d", "--downscale", type=int, default=8)
    parser.add_argument("-w", "--weights")
    parser.add_argument("--save-dir", default="./output_generated_mask_v4", type=Path)
    parser.add_argument("--margin", type=int, default=56)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--timestamp", action="store_true")

    args = parser.parse_args()
    args.slide_paths = list(map(Path, args.slide_paths))
    if args.timestamp:
        args.save_dir /= datetime.datetime.now().isoformat().replace(":", "_")

    args.save_dir.mkdir(parents=True, exist_ok=True)

    main(args)
