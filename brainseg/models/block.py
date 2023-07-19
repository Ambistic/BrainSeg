from keras.engine.base_layer import Layer
from keras.layers import Conv2D, BatchNormalization, LeakyReLU, Add


class ConvolutionBlock(Layer):
    def __init__(self, filters, size, strides=(1, 1), padding='same', activation=True, name=None):
        super(ConvolutionBlock, self).__init__(name=name)
        self.act = None
        self.bn = None
        self.conv = None
        self.filters = filters
        self.size = size
        self.strides = strides
        self.padding = padding
        self.activation = activation

    def build(self, input_shape):
        self.conv = Conv2D(self.filters, self.size, strides=self.strides, padding=self.padding)
        self.bn = BatchNormalization()
        self.act = LeakyReLU(alpha=0.1)

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        if self.activation:
            x = self.act(x)
        return x


class ResidualBlock(Layer):
    def __init__(self, num_filters=16, name=None):
        super(ResidualBlock, self).__init__(name=name)
        self.num_filters = num_filters
        self.act = LeakyReLU(alpha=0.1)
        self.bn1 = BatchNormalization()
        self.bn2 = BatchNormalization()
        self.conv1 = ConvolutionBlock(self.num_filters, (3, 3))
        self.conv2 = ConvolutionBlock(self.num_filters, (3, 3), activation=False)
        self.add = Add()

    def call(self, inputs):
        x = self.act(inputs)
        x = self.bn1(x)
        inputs = self.bn2(inputs)
        x = self.conv1(x)
        x = self.conv2(x)

        x = self.add([x, inputs])
        return x
