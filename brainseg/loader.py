from .provider import Provider, provider as _provider


class Loader:
    def __init__(self, data, provider: Provider = None, preprocess=None):
        self.data = data
        if provider is None:
            provider = _provider
        self.provider = provider
        self.preprocess = preprocess

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        args = self.image(item), self.label(item)

        if self.preprocess is not None:
            args = self.preprocess(self.data[item], *args)

        return args

    def image(self, item):
        return self.provider.image(self.data[item])

    def label(self, item):
        x = self.provider.mask(self.data[item])
        return x
