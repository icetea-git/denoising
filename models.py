import tensorflow as tf
from keras.layers import *
from keras.models import Model


def downsample(x, filter, kernel_size=(3, 3), padding='same', strides=1):
    c = Conv2D(filter, kernel_size, padding=padding, strides=strides, activation='relu')(x)
    c = Conv2D(filter, kernel_size, padding=padding, strides=strides, activation='relu')(c)
    p = MaxPool2D((2, 2), (2, 2))(c)
    return c, p


def upsample(x, skip, filter, kernel_size=(3, 3), padding='same', strides=1):
    us = UpSampling2D((2, 2))(x)
    concat = Concatenate()([us, skip])
    c = Conv2D(filter, kernel_size, padding=padding, strides=strides, activation='relu')(concat)
    c = Conv2D(filter, kernel_size, padding=padding, strides=strides, activation='relu')(c)
    return c


def bottleneck(x, filter, kernel_size=(3, 3), padding='same', strides=1):
    c = Conv2D(filter, kernel_size, padding=padding, strides=strides, activation='relu')(x)
    c = Conv2D(filter, kernel_size, padding=padding, strides=strides, activation='relu')(c)
    return c


def simple_autoencoder(training_data):
    inputs = Input(shape=training_data.shape[1:])

    x = Conv2D(32, (3, 3), activation='relu', padding='same', name='Conv1')(inputs)
    x = MaxPooling2D((2, 2), padding='same', name='pool1')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='Conv2')(x)
    x = MaxPooling2D((2, 2), padding='same', name='pool2')(x)

    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='Conv3')(x)
    x = UpSampling2D((2, 2), name='upsample1')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same', name='Conv4')(x)
    x = UpSampling2D((2, 2), name='upsample2')(x)
    x = Conv2D(1, (3, 3), activation='sigmoid', padding='same', name='Conv5')(x)

    model = Model(inputs, x)
    return model


def u_net(training_data):
    filters = [32, 64, 128, 256, 512]
    inputs = Input(shape=training_data.shape[1:])

    p0 = inputs
    d1, p1 = downsample(p0, filters[0])
    d2, p2 = downsample(p1, filters[1])
    d3, p3 = downsample(p2, filters[2])
    d4, p4 = downsample(p3, filters[3])

    bn = bottleneck(p4, filters[4])

    u1 = upsample(bn, d4, filters[3])
    u2 = upsample(u1, d3, filters[2])
    u3 = upsample(u2, d2, filters[1])
    u4 = upsample(u3, d1, filters[0])

    outputs = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(u4)
    model = Model(inputs, outputs)
    return model


def SSIM(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))
