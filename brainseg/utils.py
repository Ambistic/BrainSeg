import pickle
import matplotlib.pyplot as plt


def save_data(data, fn):
    with open(fn, "wb") as f:
        pickle.dump(data, f)


def load_data(fn):
    with open(fn, "rb") as f:
        return pickle.load(f)


def show_batch(batch):
    for i, (a, b) in enumerate(zip(*batch)):
        plt.subplot(2, len(batch[0]), 1 + i)
        plt.imshow(a)
        plt.subplot(2, len(batch[0]), 1 + i + len(batch[0]))
        plt.imshow(b, vmin=0, vmax=1)
    plt.colorbar()


def show_batch_bires(batch):
    # flatten batch
    batch = (*batch[0], batch[1])
    for i, (a, b, c) in enumerate(zip(*batch)):
        plt.subplot(3, len(batch[0]), 1 + i)
        plt.imshow(a)
        plt.subplot(3, len(batch[0]), 1 + i + len(batch[0]))
        plt.imshow(b)
        plt.subplot(3, len(batch[0]), 1 + i + len(batch[0]) * 2)
        plt.imshow(c, vmin=0, vmax=1)
    plt.colorbar()
