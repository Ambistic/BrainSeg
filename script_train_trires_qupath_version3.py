#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os

from brainseg.models.utils import transfer_weights

os.environ["SM_FRAMEWORK"] = "tf.keras"
import segmentation_models as sm
# sm.set_framework("tf.keras")
from sklearn.model_selection import train_test_split

from brainseg.generator import TrainGenerator, TestGenerator
from brainseg.loader import Loader
from brainseg.provider import provider
from brainseg.utils import show_images, load_data
from brainseg.slide_provider import MultiSlideHandler, QuPathMultiSlideHandler

import matplotlib.pyplot as plt
import numpy as np
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from tensorflow.keras.optimizers import Adam
from skimage.transform import resize
from brainseg.utils import to_color
from itertools import chain

from pathlib import Path
import os
import re

from brainseg.models.multires_unet5 import multires_unet

_SEQ = None


def get_seq_augment():
    global _SEQ
    if _SEQ is None:
        _SEQ = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.Rot90([0, 1, 2, 3]),
        ])
    return _SEQ


def adhoc_put_outline(desc):
    desc = desc.copy()
    desc["masks"][1] = desc["masks"][1].parent / "outline.png"
    return desc


def preprocess_augment(x, y):
    seq = get_seq_augment()
    current_seq = seq.to_deterministic()
    
    # we have 3 masks !!!
    x = map(np.array, x)
    y = map(np.array, y)

    if np.random.randint(0, 2):
        x = map(np.fliplr, x)
        y = map(np.fliplr, y)
    
    augmented = list(map(lambda inp: current_seq(
        image=inp[0], segmentation_maps=SegmentationMapsOnImage(inp[1], shape=inp[1].shape)
    ), zip(x, y)))
    
    x, y = zip(*augmented)
    
    x = map(lambda x_: x_ / 255., x)
    y = map(lambda y_: y_.arr.astype(np.float64), y)
    
    return list(x), list(y)


# In[10]:


def preprocess(x, y):
    x = map(np.array, x)
    y = map(np.array, y)
    
    x = map(lambda x_: x_ / 255., x)
    y = map(lambda y_: y_.astype(np.float64), y)
    # y = list(y)
    return list(x), list(y)


def no_downscale_mask(descriptor):
    desc = descriptor[1].copy()
    desc["downscales_masks"] = []
    return (descriptor[0], desc)


def prepare_data(descriptor_dir):
    sh = QuPathMultiSlideHandler()
    provider.register(sh)

    descriptor_files = list(filter(lambda x: x.endswith(".desc"), os.listdir(descriptor_dir)))
    all_descs = [load_data(descriptor_dir / desc_file) for desc_file in descriptor_files]
    flat_descs = [x for y in all_descs for x in y]

    # correct
    # flat_descs = list(map(adhoc_put_outline, flat_descs))

    dataset = [("qupath_multi", x) for x in flat_descs]

    train_dataset, test_dataset = train_test_split(dataset, random_state=0)

    test_dataset_red = [no_downscale_mask(d) for d in test_dataset]

    train_gen = TrainGenerator(Loader(train_dataset), batch_size=16, preprocess=preprocess_augment)
    test_gen = TestGenerator(Loader(test_dataset), batch_size=16, preprocess=preprocess)
    test_gen_red = TestGenerator(Loader(test_dataset_red), batch_size=16, preprocess=preprocess)

    return train_gen, test_gen, test_gen_red


def get_model(weights=None):
    model_full = multires_unet(n_res=3, n_classes=2, im_sz=224, n_channels=3,
                               n_filters_start=32, growth_factor=1.2, upconv=True,
                               all_outputs=True)

    model_red = multires_unet(n_res=3, n_classes=2, im_sz=224, n_channels=3,
                              n_filters_start=32, growth_factor=1.2, upconv=True,
                              all_outputs=False)

    model_full.compile(
        Adam(learning_rate=1e-4),
        loss=sm.losses.bce_jaccard_loss,
        metrics=[sm.metrics.iou_score],
    )

    model_red.compile(
        Adam(learning_rate=1e-4),
        loss=sm.losses.bce_jaccard_loss,
        metrics=[sm.metrics.iou_score],
    )

    if weights is not None:
        model_full.load_weights(weights)

    return model_full, model_red


def train(model_full, model_red, train_gen, test_gen, test_gen_red, starts_from, max_epochs, root_output="./default"):
    for i in range(starts_from, max_epochs):
        x, y = test_gen[0]
        show_image_from_batch((x, y), row_size=16)
        plt.savefig(f"{root_output}_ref_e{i}.png")
        y_ = model_full.predict(x)
        show_image_from_batch((x, y_), row_size=16)
        plt.savefig(f"{root_output}_pred_e{i}.png")

        model_full.fit(train_gen, use_multiprocessing=True, workers=8)
        transfer_weights(model_full, model_red)
        # evaluate on single scale
        res = model_red.evaluate(test_gen_red)
        print(res)
        # export image prediction on multi scale
        x, y = test_gen[0]
        show_image_from_batch((x, y), row_size=16)
        plt.savefig(f"{root_output}_ref_e{i}.png")
        y_ = model_full.predict(x)
        show_image_from_batch((x, y_), row_size=16)
        plt.savefig(f"{root_output}_pred_e{i}.png")

        iou = "{:.3f}".format(res[1])
        model_full.save_weights(f"{root_output}_e{i}_iou{iou}.h5")
        return  # because of the bug

    res = model_red.evaluate(test_gen)
    print('The final result is : ', res)


# for notebook visualisation
def show_image_from_batch(b, row_size=None):
    """
    Use the batch size as row_size to have the best presentation of the images
    `show_image_from_batch(batch, row_size=batch_size)
    """
    x, y = b
    flat = [i for (x_, y_) in zip(x, y) for i in chain(x_, y_)]
    plt.figure(figsize=(16, 12))
    show_images(flat, row_size=row_size)


def extract_epoch(name, root=""):
    try:
        res = int(re.findall(root + "_e(\d+)_", name)[0])
    except:
        res = None
        
    return res


def pick_last(names, extractor):
    value = -1
    kept_name = None
    
    for name in names:
        epoch = extractor(name)
        if epoch is None:
            epoch = -1
            
        if epoch > value:
            value, kept_name = epoch, name
            
    return value, kept_name


def main(descriptor_dir, directory, root_name, max_epochs, weights):
    names = os.listdir(directory)
    last_epoch, name_weights = pick_last(names, lambda x: extract_epoch(x, root_name))
    start_from = last_epoch + 1

    print(last_epoch)
    model_full, model_red = get_model(weights)
    if name_weights is not None:
        model_full.load_weights(os.path.join(directory, name_weights))

    train_gen, test_gen, test_gen_red = prepare_data(descriptor_dir)
    root_output = os.path.join(directory, root_name)
    train(model_full, model_red, train_gen, test_gen, test_gen_red, start_from, max_epochs, root_output)


if __name__ == "__main__":
    """This script automatically run an epoch starting from the previous model generated"""
    directory = "/media/tower/LaCie/Data/models/trires/"
    root_name = "model_test_vSep23"
    max_epochs = 40
    descriptor_dir = Path("/srv/share/descriptors/v2/")
    weights = "/media/tower/LaCie/Data/models/trires/model_test_vJun23_e15_iou0.943.h5"
    main(descriptor_dir, directory, root_name, max_epochs, weights)
