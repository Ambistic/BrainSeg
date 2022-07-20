# u-net model with up-convolution or up-sampling and weighted binary-crossentropy as loss func

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, \
    Conv2DTranspose, BatchNormalization, Dropout, LeakyReLU, Add, Cropping2D


def convolution_block(x, filters, size, strides=(1, 1), padding='same', activation=True):
    x = Conv2D(filters, size, strides=strides, padding=padding)(x)
    x = BatchNormalization()(x)
    if activation:
        x = LeakyReLU(alpha=0.1)(x)
    return x


def residual_block(blockInput, num_filters=16):
    x = LeakyReLU(alpha=0.1)(blockInput)
    x = BatchNormalization()(x)
    blockInput = BatchNormalization()(blockInput)
    x = convolution_block(x, num_filters, (3, 3))
    x = convolution_block(x, num_filters, (3, 3), activation=False)
    x = Add()([x, blockInput])
    return x


def get_lowres_encoder(im_sz=224, n_channels=3, n_filters_start=32, growth_factor=1.2, power_downscale=1):
    droprate = 0.1
    n_filters = n_filters_start
    inputs = Input((im_sz, im_sz, n_channels))

    conv1 = convolution_block(inputs, n_filters, (3, 3))
    conv1 = residual_block(conv1, n_filters)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool1 = Dropout(droprate)(pool1)

    n_filters *= growth_factor
    pool1 = BatchNormalization()(pool1)
    conv2 = convolution_block(pool1, n_filters, (3, 3))
    conv2 = residual_block(conv2, n_filters)
    conv2 = residual_block(conv2, n_filters)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool2 = Dropout(droprate)(pool2)

    n_filters *= growth_factor
    pool2 = BatchNormalization()(pool2)
    pool2 = convolution_block(pool2, n_filters, (3, 3))
    conv3 = residual_block(pool2, n_filters)
    conv3 = residual_block(conv3, n_filters)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    pool3 = Dropout(droprate)(pool3)

    n_filters *= growth_factor
    pool3 = BatchNormalization()(pool3)
    pool3 = convolution_block(pool3, n_filters, (3, 3))
    conv4_0 = residual_block(pool3, n_filters)
    conv4_0 = residual_block(conv4_0, n_filters)

    up = conv4_0
    for _ in range(power_downscale):
        print("Downscaling", _)
        up = Cropping2D(cropping=(7, 7))(up)
        up = UpSampling2D(size=(2, 2))(up)

    pool4_0 = MaxPooling2D(pool_size=(2, 2))(up)
    pool4_0 = Dropout(droprate)(pool4_0)

    n_filters *= growth_factor
    pool4_0 = BatchNormalization()(pool4_0)
    pool4_0 = convolution_block(pool4_0, n_filters, (3, 3))
    conv5 = residual_block(pool4_0, n_filters)
    conv5 = residual_block(conv5, n_filters)
    conv5 = Dropout(droprate)(conv5)  # dim is 28x28

    output = conv5

    model = Model(inputs=inputs, outputs=output)
    return model


def multires_unet(n_res=2, n_classes=1, im_sz=224, n_channels=3, n_filters_start=32, growth_factor=1.2, upconv=True):
    droprate = 0.1
    n_filters = n_filters_start
    lowres_encoders = [get_lowres_encoder(
        im_sz=im_sz, n_channels=n_channels, n_filters_start=n_filters_start,
        growth_factor=growth_factor, power_downscale=i + 1
    ) for i in range(n_res - 1)]

    inputs = Input((im_sz, im_sz, n_channels))

    conv1 = convolution_block(inputs, n_filters, (3, 3))
    conv1 = residual_block(conv1, n_filters)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool1 = Dropout(droprate)(pool1)

    n_filters *= growth_factor
    pool1 = BatchNormalization()(pool1)
    conv2 = convolution_block(pool1, n_filters, (3, 3))
    conv2 = residual_block(conv2, n_filters)
    conv2 = residual_block(conv2, n_filters)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool2 = Dropout(droprate)(pool2)

    n_filters *= growth_factor
    pool2 = BatchNormalization()(pool2)
    pool2 = convolution_block(pool2, n_filters, (3, 3))
    conv3 = residual_block(pool2, n_filters)
    conv3 = residual_block(conv3, n_filters)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    pool3 = Dropout(droprate)(pool3)

    n_filters *= growth_factor
    pool3 = BatchNormalization()(pool3)
    pool3 = convolution_block(pool3, n_filters, (3, 3))
    conv4_0 = residual_block(pool3, n_filters)
    conv4_0 = residual_block(conv4_0, n_filters)
    pool4_0 = MaxPooling2D(pool_size=(2, 2))(conv4_0)
    pool4_0 = Dropout(droprate)(pool4_0)

    n_filters *= growth_factor
    pool4_0 = BatchNormalization()(pool4_0)
    pool4_0 = convolution_block(pool4_0, n_filters, (3, 3))
    conv5 = residual_block(pool4_0, n_filters)
    conv5 = residual_block(conv5, n_filters)
    conv5 = Dropout(droprate)(conv5)

    conv5 = concatenate([conv5] + [lowres_encoder.output for lowres_encoder in lowres_encoders])

    # INCREASING SIZE

    n_filters //= growth_factor
    if upconv:
        up6_2 = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(conv5), conv4_0])
    else:
        up6_2 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4_0])
    up6_2 = BatchNormalization()(up6_2)
    up6_2 = convolution_block(up6_2, n_filters, (3, 3))
    conv6_2 = residual_block(up6_2, n_filters)
    conv6_2 = residual_block(conv6_2, n_filters)
    conv6_2 = Dropout(droprate)(conv6_2)

    n_filters //= growth_factor
    if upconv:
        up7 = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(conv6_2), conv3])
    else:
        up7 = concatenate([UpSampling2D(size=(2, 2))(conv6_2), conv3])
    up7 = BatchNormalization()(up7)
    up7 = convolution_block(up7, n_filters, (3, 3))
    conv7 = residual_block(up7, n_filters)
    conv7 = residual_block(conv7, n_filters)
    conv7 = Dropout(droprate)(conv7)

    n_filters //= growth_factor
    if upconv:
        up8 = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(conv7), conv2])
    else:
        up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2])
    up8 = BatchNormalization()(up8)
    up8 = convolution_block(up8, n_filters, (3, 3))
    conv8 = residual_block(up8, n_filters)
    conv8 = residual_block(conv8, n_filters)
    conv8 = Dropout(droprate)(conv8)

    n_filters //= growth_factor
    if upconv:
        up9 = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(conv8), conv1])
    else:
        up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1])
    up9 = BatchNormalization()(up9)
    up9 = convolution_block(up9, n_filters, (3, 3))
    conv9 = residual_block(up9, n_filters)
    conv9 = residual_block(conv9, n_filters)

    conv10 = Conv2D(n_classes, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs] + [lowres_encoder.inputs for lowres_encoder in lowres_encoders],
                  outputs=conv10)

    return model
