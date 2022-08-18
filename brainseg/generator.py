from tensorflow.keras.utils import Sequence
from .provider import provider
from random import shuffle
from math import floor, ceil
import numpy as np


# name is not explicit for arrayisation
def swapaxis(x):
    # from list sample of list input to list input of list sample
    return list(map(np.array, zip(*x)))


class Generator(Sequence):
    def __init__(
            self,
            data,
            batch_size=32,
            preprocess=None,
            ):
        self.data = data
        self.preprocess = preprocess
        self.batch_size = batch_size
        self.indexes = list(range(len(self.data)))

    def __data_generation(self, batch):
        x, y = [], []

        for index in batch:
            x_, y_ = self.data[index]
            if self.preprocess is not None:
                x_, y_ = self.preprocess(x_, y_)
            x.append(x_)
            y.append(y_)

        return x, y

    def __getitem__(self, index):
        if index >= len(self):
            return None

        batch = self.indexes[
            index * self.batch_size: (index + 1) * self.batch_size
        ]

        # Generate data
        x, y = self.__data_generation(batch)

        # handle if format is list
        if isinstance(x[0], list):
            x = swapaxis(x)
        else:
            x = np.array(x)

        if isinstance(y[0], list):
            y = swapaxis(y)
        else:
            y = np.array(y)

        return x, y


class TrainGenerator(Generator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.on_epoch_end()

    def __len__(self):
        return floor(len(self.data) / self.batch_size)

    def on_epoch_end(self):
        shuffle(self.indexes)


class TestGenerator(Generator):
    def __len__(self):
        return ceil(len(self.data) / self.batch_size)


class InfiniteGenerator(Generator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.on_epoch_end()

    # still has a len
    def __len__(self):
        return floor(len(self.data) / self.batch_size)

    def on_epoch_end(self):
        shuffle(self.indexes)

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n >= len(self):
            self.n = 0
            self.on_epoch_end()
        ret = self[self.n]
        self.n += 1
        return ret
