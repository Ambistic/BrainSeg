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

from brainseg.slide_provider_top_left import MultiSlideMaskHandler

sm.set_framework("tf.keras")

MODEL = None


def get_model(args):
    global MODEL

    if MODEL is None:
        # ONLY THIS LINE HAS TO BE CHANGED
        model = multires_unet(n_res=2, n_classes=1, im_sz=224, n_channels=3, n_filters_start=32,
                              growth_factor=1.2, all_outputs=True)
        model.load_weights(args.weights)
        model.compile()
        MODEL = model

    return MODEL


def segment(args, slide_path, x, y, size):
    model = get_model(args)
    image = provider.image(("multi_slide_mask", dict(
        slide_name=slide_path.name,
        downsample=args.downscale,
        downscales=[32],
        origin=(x, y),
        size=size,
    )))
    image = list(map(lambda im: np.array([np.asarray(im) / 255.]), image))
    # the second [0] is because the model has 2 outputs
    mask = model.predict(image, verbose=False)[0][0]

    # no inversion for this model
    # mask = 1 - mask

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
    handler = MultiSlideMaskHandler(args.slide_paths, "", "mask")
    provider.register(handler)


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
    full_mask = np.zeros(full_size, dtype=np.uint8)

    print("Running segmentation")
    for x, y in tqdm(list(generate_coords(args.size, args.margin, full_size))):
        mask = segment(args, slide_path, x * args.downscale, y * args.downscale, args.size)
        add_patch(full_mask, mask, x, y, args.size, args.margin)

    print("Saving mask")
    save_mask(full_mask.transpose(), args.save_dir / (slide_path.stem + ".png"))


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

    args = parser.parse_args()
    args.slide_paths = list(map(Path, args.slide_paths))

    main(args)
