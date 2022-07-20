import argparse
from functools import partial
from pathlib import Path

import segmentation_models as sm
from sklearn.model_selection import train_test_split

from brainseg.generator import TrainGenerator, TestGenerator
from brainseg.loader import Loader
from brainseg.models.bires_unet import bi_res_unet
from brainseg.utils import load_data, show_batch, to_color, rgb_to_multi
from brainseg.image_provider import ImageHandler, BiResImageHandler
from brainseg.provider import provider
from brainseg.streamlit.manager import list_all

import matplotlib.pyplot as plt
import numpy as np
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from tensorflow.keras.optimizers import Adam
from skimage.transform import resize

sm.set_framework("tf.keras")


def get_model(args):
    if args.model == "bires":
        model = bi_res_unet(n_classes=args.n_classes, im_sz=args.size, n_channels=args.channel)
    else:
        raise ValueError("Unknown model")

    model.compile(
        Adam(learning_rate=args.lr),
        loss=sm.losses.bce_jaccard_loss,
        metrics=[sm.metrics.iou_score],
    )
    return model


def get_handler(args):
    if args.model == "bires":
        return BiResImageHandler()


def get_dataset(args):
    dataset = list_all(args.root, min_threshold=10)

    nb_slides = len(set([x[1]["data_name"][:14] for x in dataset]))
    print(f"Found {len(dataset)} patches from {nb_slides} slides")

    return train_test_split(dataset, random_state=0)


seq = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
    iaa.Rot90([0, 1, 2, 3]),
])


def preprocess_augment(args, x, y):
    x = np.asarray(x)
    y = np.asarray(y)  # TODO make conversion
    y = rgb_to_multi(y, args.color)

    seg = SegmentationMapsOnImage(y, shape=y.shape)
    x, y = seq(image=x, segmentation_maps=seg)
    x = x / 255.
    y = y / 255.

    y = y.astype(np.float64)
    return x, y


def preprocess(args, x, y):
    x = np.asarray(x) / 255.
    y = np.asarray(y) / 255.
    y = y.astype(np.float64)
    return x, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--root", type=Path)
    parser.add_argument("-o", "--output", type=Path)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--model", type=str, default="bires")
    parser.add_argument("-a", "--area", help="Area to use", nargs="+")

    args = parser.parse_args()

    args.color = to_color(args.area)

    sh = get_handler(args)
    provider.register(sh)

    model = get_model(args)
    train_dts, test_dts = get_dataset(args)

    train_gen = TrainGenerator(Loader(train_dts), batch_size=4, preprocess=partial(preprocess_augment, args))
    test_gen = TestGenerator(Loader(test_dts), batch_size=4, preprocess=partial(preprocess, args))


if __name__ == "__main__":
    main()
