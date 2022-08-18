from abc import ABC, abstractmethod

import brainseg.streamlit.load


class DataHandler(ABC):
    name = None

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def load_image(self, element):
        pass

    @abstractmethod
    def load_mask(self, element):
        pass


class Provider:
    def __init__(self):
        self.datasets = dict()
        self.handlers = dict()

    def register(self, handler: DataHandler):
        assert isinstance(handler, DataHandler)
        self.handlers[handler.name] = handler

    def image(self, item):
        dataset, element = item
        return self.handlers[dataset].load_image(element)

    def mask(self, item):
        dataset, element = item
        return self.handlers[dataset].load_mask(element)

    def __getitem__(self, item):
        return self.image(item), self.mask(item)


provider = Provider()
