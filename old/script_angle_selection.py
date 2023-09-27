"""
TODO :
- investigate the error
- implement the coord mapping
- Add final printing of `t` and `s` as used by histology
- Change the preprocess to welcome the new best models

"""
import argparse
import os.path
from math import ceil
from pathlib import Path

import tensorflow as tf

from brainseg.config import fill_with_config
from brainseg.path import build_path_histo

for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)


from brainseg.models import siamese
from tensorflow.keras import optimizers

from brainseg.handlers.siamese_histo_mri import SingleImageHandler
from brainseg.provider import provider
from brainseg.loader import Loader
from brainseg.generator import X0TestGenerator
import pandas as pd
import numpy as np
from skimage.transform import resize
import re
from brainseg.image import resize_and_pad_center

from sklearn.metrics import mean_squared_error as mse
from itertools import product
import nibabel as nib

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


OUTPUT_SHAPE = (256, 256, 3)
_EMB_MRI_CACHE = dict()


def wb_coord_mri_to_index_mri_ap(x):
    return 128 + 2 * x


def index_mri_to_wb_coord_ap(x):
    return (x - 128) / 2


def wb_coord_mri_to_histo_coord(x, t, s):
    return t + s * x


def histo_coord_to_wb_coord_mri(x, t, s):
    return (x - t) / s


# Because of the czi bug when exporting from zen or using python czifile library
def swap_channels_bug_czi(image):
    # Swap the 2nd and 3rd channels
    image_swapped = np.copy(image)
    image_swapped[:, :, 0], image_swapped[:, :, 2] = image[:, :, 2], image[:, :, 0]
    return image_swapped


def preprocess_image_histo(image):
    size = image.shape[:2]
    # because it's normally divided by 128 but at this point it's already divided by 10
    target_size = (ceil(size[0] / 12.8), ceil(size[1] / 12.8))

    image = resize(image, target_size)
    image = resize_and_pad_center(image, 512, 512, background=255)

    return image


def get_model(output_shape, weights_path):
    siamese_network = siamese.get_siamese_model_uniform(output_shape)
    siamese_model = siamese.SiameseModel(siamese_network)
    siamese_model.compile(optimizer=optimizers.Adam(0.0001))
    siamese_model.built = True  # cheat
    siamese_model.load_weights(weights_path)

    embedder = siamese.get_embedder(output_shape, 1)

    embedder.set_weights(siamese_model.layers[0].layers[3].get_weights())
    return embedder


def prepro_spe(x):
    if x.ndim == 2 or (x.ndim == 3 and x.shape[2] == 1):  # proxy for "is mri"
        x = resize_and_pad_center(x).reshape((256, 256, 1))
        x = np.rot90(x, 1)
        x = np.repeat(x, 3, axis=2)

    elif x.ndim == 3 and x.shape[2] >= 3:  # proxy for "is histo"
        x = preprocess_image_histo(x)
        x = swap_channels_bug_czi(x)
        x = resize(x, (256, 256))
    return x


def preprocess_siamese(x):
    x = np.asarray(x)
    # x = seq(image=x)
    x = (x - x.mean()) / max(x.std(), 1e-4)
    x = prepro_spe(x)
    return x


def standardize_brightness_no_background(x, bg_value=0, p=90):
    x_flat = x.flatten()
    if np.all(np.floor(x_flat) == bg_value):
        return x
    thr = np.percentile(x_flat[np.floor(x_flat) != bg_value], p)
    return np.clip(x * 255. / thr, 0, 255)


def prepro_spe_noaugment(x):
    if x.ndim == 2 or (x.ndim == 3 and x.shape[2] == 1):  # proxy for "is mri"
        x = resize(x, (int(x.shape[0] * 1.8), int(x.shape[1] * 1.8)))
        x = x[50:-50, 70:-30]
        x = standardize_brightness_no_background(x, p=100)
        # x = resize(x, (int(x.shape[0] * 1.6), int(x.shape[1] * 1.6)))
        # x = x[30:-30, 30:-30]
        # x = standardize_brightness_no_background(x, p=100)
        x = resize_and_pad_center(x).reshape((256, 256, 1))
        x = np.rot90(x, 1)
        x = np.repeat(x, 3, axis=2)
        x = x / 255.
        x = (x - x.mean()) / max(x.std(), 1e-4)
    elif x.ndim == 3 and x.shape[2] >= 3:  # proxy for "is histo"
        x = preprocess_image_histo(x)
        x = swap_channels_bug_czi(x)
        x = standardize_brightness_no_background(x, bg_value=255)
        x = resize(x, (256, 256))
        x = x / 255.
        x = (x - x.mean()) / max(x.std(), 1e-4)
    return x


def preprocess_siamese_noaugment(x):
    x = np.asarray(x)
    x = prepro_spe_noaugment(x)
    return x


def dictify_embeddings(descriptors, embeddings):
    return {int(d[1]["slide_id"]): emb for d, emb in zip(descriptors, embeddings)}


def extract_mri_name(name):
    return re.findall(r".*([MC]\d{3}).*", name)[0]


def list_histo_descriptors(args):
    descriptors = []
    # for slice_id in range(args.start, args.end, args.step):
    for slice_id in [63, 65, 69, 71, 75, 77, 81, 87, 89, 93, 95, 99, 101, 103, 105, 107,
                     109, 111, 113, 117, 123, 125, 129, 131, 135, 137, 141, 145, 147, 149,
                     153, 155, 159, 161, 165, 167, 171, 177, 179, 181, 183, 187, 197, 199,
                     201, 205, 207, 211, 213, 215, 217, 219, 221, 223, 227, 229, 231, 233,
                     239, 241, 243, 245, 247, 249, 251, 253, 257, 259, 261, 263, 265, 267,
                     271, 273, 275, 277, 279, 283, 285, 287, 289, 291, 295, 297, 299, 301,
                     309, 317, 319, 321, 323]:
        slice_file = build_path_histo(
            args.histo_dir, slice_id, args.histo_mask
        )

        if not os.path.exists(slice_file):
            continue

        desc = dict(
            image_name=slice_file,
            monkey_name="NA",
            slide_id=slice_id,
        )
        descriptors.append(desc)
    return descriptors


def get_embeddings_histo(args, embedder, handler_name):
    descriptors_self = list_histo_descriptors(args)
    for desc in descriptors_self:
        desc["type"] = "histo"
    descriptors_self = list(map(lambda x: (handler_name, x), descriptors_self))
    if len(descriptors_self) == 0:
        raise ValueError("There are no input data, check the `histo_dir` is not empty and the `histo_mask` "
                         "if correct.")

    test_loader = Loader(descriptors_self)
    test_gen = X0TestGenerator(test_loader, batch_size=4, preprocess=preprocess_siamese_noaugment)

    embeddings = embedder.predict(test_gen, verbose=False)

    return dictify_embeddings(descriptors_self, embeddings)


def list_descriptors_mri_only(args):
    # for each mri file
    # cut -20 and +20, and generate a "slide" per index
    descriptors = []

    nib_img = nib.load(args.mri_brain_file)
    size = nib_img.get_fdata().shape[1]

    for index in range(+20, size - 20):
        desc = dict(
            mri_name=args.mri_brain_file,
            monkey_name="NA",
            slide_id=index,  # for triplet construction
            rotx=0,
            rotz=0,
            mri_id=index,
            has_mask=False,  # for mask switch (oasis in training data)
        )
        descriptors.append(desc)

    return descriptors


def get_embeddings_mri(args, embedder, data_name, rotx, rotz):
    global _EMB_MRI_CACHE
    key = (rotx, rotz)
    if key in _EMB_MRI_CACHE:
        return _EMB_MRI_CACHE[key]
    descriptors_mri = list_descriptors_mri_only(args)

    for desc in descriptors_mri:
        desc["type"] = "mri"
        desc["rotx"] = rotx
        desc["rotz"] = rotz

    descriptors_mri = list(map(lambda x: (data_name, x), descriptors_mri))

    test_loader = Loader(descriptors_mri)
    test_gen = X0TestGenerator(test_loader, batch_size=4, preprocess=preprocess_siamese_noaugment)

    embeddings = embedder.predict(test_gen, verbose=False)
    res = dictify_embeddings(descriptors_mri, embeddings)

    _EMB_MRI_CACHE[key] = res
    return res


def evaluate_embeddings(emb_histo, emb_mri, y, s):
    # here we fix the s
    total_comparison = 0
    total_score = 0
    scores = []
    # print(min(emb_histo.keys()), max(emb_histo.keys()))
    for index, value_histo in emb_histo.items():
        # mri_id = int(y - s * index)  # if axis is reversed in histo compared to mri
        wb_mri_id = histo_coord_to_wb_coord_mri(index, t=y, s=s)
        mri_index = wb_coord_mri_to_index_mri_ap(wb_mri_id)
        mri_index = int(np.round(mri_index))
        # print(f"match histo {index} with mri {mri_index}")
        # mri_id = int(y + s * index)  # if axis is same in histo compared to mri
        if mri_index not in emb_mri:
            continue
        value_mri = emb_mri.get(mri_index)

        try:
            mean_square_error = mse(value_mri, value_histo)
        except:
            print("Bug")
        else:
            total_score += mean_square_error
            total_comparison += 1
            scores.append(mean_square_error)

    if total_comparison == 0:
        return 1e9, 0

    robust_mean_1 = np.mean(sorted(scores)[int(len(scores) * 0.1):-int(len(scores) * 0.1)]) * 10
    robust_mean_2 = np.mean(sorted(scores)[int(len(scores) * 0.2):-int(len(scores) * 0.2)]) * 10
    robust_mean_3 = np.mean(sorted(scores)[int(len(scores) * 0.3):-int(len(scores) * 0.3)]) * 10

    print(f"debugging predictions {y} {s}, min {np.min(scores) * 10:.4f}, max {np.max(scores) * 10:.4f}, "
          f"mean {np.mean(scores) * 10:.4f}, "
          f"p10 {np.percentile(scores, 10) * 10:.4f}, p90 {np.percentile(scores, 90) * 10:.4f}, "
          f"median {np.median(scores) * 10:.4f}, robust_means {robust_mean_1:.4f} {robust_mean_2:.4f} "
          f"{robust_mean_3:.4f}")

    # return total_score / total_comparison, total_comparison / len(emb_histo)
    # median could maybe do it also
    # return robust_mean_1 + np.percentile(scores, 10) * 50, total_comparison / len(emb_histo)
    return np.mean(scores) * 10, total_comparison / len(emb_histo)


def initial_guess(histo, mri, s):
    trials = []
    for y in range(0, 400, 20):
        val, ratio = evaluate_embeddings(histo, mri, y, s)
        print("ratio", ratio)
        if ratio != 1.0:
            val = 1e9
        trials.append(dict(
            y=y, s=s, val=val))
    df_res = pd.DataFrame(trials)

    return df_res.sort_values("val").iloc[0]


def update_guess(histo, mri, y0, s):
    trials = []
    for y, s_ in product(np.arange(y0 - 6, y0 + 6, 1), np.arange(s - 0.5, s + 0.5, 0.1)):
        val, ratio = evaluate_embeddings(histo, mri, y, s_)
        if ratio != 1.0:
            val = 1e9
        trials.append(dict(
            y=y, s=s_, val=val))
    df_res = pd.DataFrame(trials)

    print(df_res)

    return df_res.sort_values("val").iloc[0]


def evaluate_x_z(args, histo_embeddings, embedder, data_name, x, z, y, s):
    print(f"Angles {x} {z}")
    mri_embeddings = get_embeddings_mri(args, embedder, data_name,
                                        rotx=x, rotz=z)
    score = evaluate_embeddings(histo_embeddings, mri_embeddings, y, s)
    return score


def update_guess_x_z(args, histo_embeddings, embedder, data_name, x0, z0, y, s):
    trials = []
    for x, z in product(list(np.arange(x0 - 3, x0 + 3, 1)), list(np.arange(z0 - 3, z0 + 3, 1))):
        val, ratio = evaluate_x_z(args, histo_embeddings, embedder, data_name,
                                  x, z, y, s)
        if ratio != 1.0:
            val = 1e9
        trials.append(dict(
            x=x, z=z, val=val))
    df_res = pd.DataFrame(trials)

    print(df_res)

    return df_res.sort_values("val").iloc[0]


def update_guess_all(args, histo_embeddings, embedder, data_name, x0, z0, y0, s0):
    trials = []
    for x, z, y, s in product(
            list(np.arange(x0 - 3, x0 + 3, 1)), list(np.arange(z0 - 3, z0 + 3, 1)),
            list(np.arange(y0 - 6, y0 + 6, 1)), list(np.arange(s0 - 0.5, s0 + 0.5, 0.1))):
        val, ratio = evaluate_x_z(args, histo_embeddings, embedder, data_name,
                                  x, z, y, s)
        if ratio != 1.0:
            val = 1e9
        trials.append(dict(
            x=x, z=z, y=y, s=s, val=val))
    df_res = pd.DataFrame(trials)

    print(df_res)

    return df_res.sort_values("val").iloc[0]


def main(args):
    shmh = SingleImageHandler()
    provider.register(shmh)

    print("Loading model")
    embedder = get_model(OUTPUT_SHAPE, args.siamese_weights)
    print("Computing histology embeddings")
    histo_dict_embeddings = get_embeddings_histo(args, embedder, shmh.name)
    print("Computing MRI embeddings for angle (0, 0)")
    mri_dict_embeddings = get_embeddings_mri(args, embedder, shmh.name, 0, 0)

    # s if this one because definition of the scale is the inverse in later processing
    d0 = initial_guess(histo_dict_embeddings, mri_dict_embeddings, s=args.scale_y)

    print("Initial guess", d0.to_dict())

    d = d0
    for i in range(5):
        d = update_guess(histo_dict_embeddings, mri_dict_embeddings, y0=d["y"], s=d["s"])

    print("Updated translation guess", d.to_dict())

    d_current = d
    res_x_z = dict(x=0, z=0)
    for i in range(5):
        best = update_guess_all(args, histo_dict_embeddings, embedder, shmh.name, x0=res_x_z["x"], z0=res_x_z["z"],
                                y0=d_current["y"], s0=d_current["s"])
        d_current["y"] = best["y"]
        d_current["s"] = best["s"]
        res_x_z["x"] = best["x"]
        res_x_z["z"] = best["z"]

    for i in range(1):
        res_x_z = update_guess_x_z(args, histo_dict_embeddings, embedder,
                                   shmh.name, res_x_z["x"], res_x_z["z"], y=d_current["y"], s=d_current["s"])
        mri_emb = get_embeddings_mri(args, embedder, shmh.name, rotx=res_x_z["x"], rotz=res_x_z["z"])
        print(f"Angles for loop {i}", res_x_z.to_dict())
        d_current = update_guess(histo_dict_embeddings, mri_emb,
                                 y0=d_current["y"], s=d_current["s"])
        print(f"Translation for loop {i}", d_current.to_dict())

    print("You can fill your config with the following :\n"
          f"translation_y = {d_current['y']}\n"
          f"scale_y = {d_current['s']}\n"
          f"angle_x = {res_x_z['x']}\n"
          f"angle_z = {res_x_z['z']}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # no config, individual operation
    parser.add_argument("-c", "--config", type=Path, default=Path("/media/tower/LaCie/REGISTRATION_PROJECT/"
                                                                  "TMP_PIPELINE_M148/config.ini"))
    parser.add_argument("--histo_dir", type=str, default=None)
    parser.add_argument("--histo_mask", type=str, default=None)
    parser.add_argument("--mri_brain_file", type=str, default=None)
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--scale_y", type=float, default=None)
    parser.add_argument("--siamese_weights", type=str, default=None)

    args_ = fill_with_config(parser)

    main(args_)
