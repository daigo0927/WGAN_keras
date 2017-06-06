# coding:utf-8

import os, sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import h5py

import keras.backend as K
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Flatten, Dropout, Activation, Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D, Conv2D

def generator(image_size = 64):

    L = int(image_size)

    inputs = Input(shape = (100, ))
    x = Dense(1024)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dense(128*int(L/8)*int(L/8)*3)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Reshape((int(L/8), int(L/8), 128*3))(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64*3, (5, 5), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32*3, (5, 5), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(1*3, (5, 5), padding = 'same')(x)
    images = Activation('tanh')(x)

    model = Model(inputs = inputs, outputs = images)

    model.summary()

    return model

def discriminator(image_size = 64):

    L = int(image_size)

    images = Input(shape = (L, L, 3))
    x = Conv2D(64, (5, 5), strides = (2, 2), padding = 'same')(images)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(128, (5, 5), strides = (2, 2), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Flatten()(x)
    x = Dense(256)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.5)(x)
    x = Dense(1)(x)
    outputs = Activation('linear')(x)

    model = Model(inputs = images, outputs = outputs)

    model.summary()

    return model
