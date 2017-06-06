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
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D, Conv2D

def generator(image_size = 64):

    L = int(image_size)

    inputs = Input(shape = (100, ))
    x = Dense(512*int(L/16)**2)(inputs) #shape(512*(L/16)**2,)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Reshape((int(L/16), int(L/16), 512))(x) # shape(L/16, L/16, 512)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(256, (5, 5), padding = 'same')(x) # shape(L/8, L/8, 256)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(128, (5, 5), padding = 'same')(x) # shape(L/4, L/4, 128)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (5, 5), padding = 'same')(x) # shape(L/2, L/2, 64)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(1*3, (5, 5), padding = 'same')(x) # shape(L, L, 3)
    images = Activation('tanh')(x)

    model = Model(inputs = inputs, outputs = images)

    model.summary()

    return model

def discriminator(image_size = 64):

    L = int(image_size)

    images = Input(shape = (L, L, 3)) 
    x = Conv2D(32, (5, 5), strides = (2, 2), padding = 'same')(images) # shape(L/2, L/2, 32)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(64, (5, 5), strides = (2, 2), padding = 'same')(x) # shape(L/4, L/4, 64)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(128, (5, 5), strides = (2, 2), padding = 'same')(x) # shape(L/8, L/8, 128)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(256, (5, 5), strides = (2, 2), padding = 'same')(x) # shape(L/16, L/16, 256)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(1, (5, 5), strides = (2, 2), padding = 'same')(x)
    outputs = AveragePooling2D()(x)

    model = Model(inputs = images, outputs = outputs)

    model.summary()

    return model
