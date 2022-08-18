# u-net model with up-convolution or up-sampling and weighted binary-crossentropy as loss func

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, \
    Conv2DTranspose, BatchNormalization, Dropout, Cropping2D

from brainseg.models.block import ConvolutionBlock, ResidualBlock


def lower_res_unet(log_downscale, n_classes=1, im_sz=224, n_channels=3, n_filters_start=32,
                   growth_factor=1.2, upconv=True):
    """

    :param log_downscale: The power of 2 of the downscale, for example if having a reference of 8 and a lower scale
    of 32, then the log_downscale is log2(32 / 8) = 2
    :param n_classes:
    :param im_sz:
    :param n_channels:
    :param n_filters_start:
    :param growth_factor:
    :param upconv:
    :return:
    """
    droprate = 0.1
    depth = 5
    n_filters = n_filters_start

    n_filters_layers = [int(n_filters_start * growth_factor ** i)
                        for i in range(depth)]
    conv_layers = []

    inputs = Input((im_sz, im_sz, n_channels))
    last = inputs

    for i in range(depth):
        conv = ConvolutionBlock(n_filters_layers[i], (3, 3))(last)
        conv = ResidualBlock(n_filters_layers[i])(conv)
        conv = ResidualBlock(n_filters_layers[i])(conv)
        conv_layers.append(conv)

        if i != depth - 1:
            last = MaxPooling2D(pool_size=(2, 2))(conv)
            last = Dropout(droprate)(last)
            last = BatchNormalization()(last)

    # INCREASING SIZE
    escaped_layer = None

    for down, i in enumerate(reversed(range(depth - 1))):
        if upconv:
            up = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(last), conv_layers[i]])
        else:
            up = concatenate([UpSampling2D(size=(2, 2))(last), conv_layers[i]])
        up = BatchNormalization()(up)
        up = ConvolutionBlock(n_filters_layers[i], (3, 3))(up)
        conv = ResidualBlock(n_filters_layers[i])(up)
        conv = ResidualBlock(n_filters_layers[i])(conv)
        if down + 1 == log_downscale:
            escaped_layer = conv
        if i != 0:
            conv = Dropout(droprate)(conv)
        last = conv

    crop_size = 7 * (2**log_downscale - 1)
    cropped_layer = Cropping2D(cropping=(crop_size, crop_size))(escaped_layer)

    conv10 = Conv2D(n_classes, (1, 1), activation='sigmoid')(last)

    return inputs, cropped_layer, conv10


def multires_unet(n_res=2, n_classes=1, im_sz=224, n_channels=3, n_filters_start=32,
                  growth_factor=1.2, upconv=True, all_outputs=False):
    droprate = 0.1
    depth = 5
    n_filters = n_filters_start
    low_models = [lower_res_unet(
        im_sz=im_sz, n_channels=n_channels, n_filters_start=n_filters_start,
        growth_factor=growth_factor, log_downscale=(2 + 2 * i)
    ) for i in range(n_res - 1)]
    low_inputs, low_emb, low_outputs = map(list, zip(*low_models))

    n_filters_layers = [int(n_filters_start * growth_factor ** i)
                        for i in range(depth)]
    conv_layers = []

    inputs = Input((im_sz, im_sz, n_channels))
    last = inputs

    for i in range(depth):
        conv = ConvolutionBlock(n_filters_layers[i], (3, 3))(last)
        conv = ResidualBlock(n_filters_layers[i])(conv)
        conv = ResidualBlock(n_filters_layers[i])(conv)
        conv_layers.append(conv)

        if i != depth - 1:
            last = MaxPooling2D(pool_size=(2, 2))(conv)
            last = Dropout(droprate)(last)
            last = BatchNormalization()(last)

    last = concatenate([last] + low_emb)

    # INCREASING SIZE

    for i in reversed(range(depth - 1)):
        if upconv:
            up = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(last), conv_layers[i]])
        else:
            up = concatenate([UpSampling2D(size=(2, 2))(last), conv_layers[i]])
        up = BatchNormalization()(up)
        up = ConvolutionBlock(n_filters_layers[i], (3, 3))(up)
        conv = ResidualBlock(n_filters_layers[i])(up)
        conv = ResidualBlock(n_filters_layers[i])(conv)
        if i != 0:
            conv = Dropout(droprate)(conv)
        last = conv

    conv10 = Conv2D(n_classes, (1, 1), activation='sigmoid')(last)

    if all_outputs:
        model = Model(inputs=[inputs] + low_inputs,
                      outputs=conv10 + low_outputs)
    else:
        model = Model(inputs=[inputs] + low_inputs,
                      outputs=conv10)

    return model
