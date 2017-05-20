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
from keras.optimizers import Adam, RMSprop

from misc.utils import combine_images

def GeneratorModel(image_size = (32, 32)):

    L = int(image_size[0])

    inputs = Input(shape = (100, ))
    x = Dense(1024)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dense(128*int(L/4)*int(L/4)*3)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Reshape((int(L/4), int(L/4), 128*3))(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64*3, (5, 5), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(1*3, (5, 5), padding = 'same')(x)
    images = Activation('tanh')(x) # output 32*32*3 images

    model = Model(inputs = inputs, outputs = images)

    model.summary()

    return model

def CriticModel(image_size = (32, 32)):

    L = int(image_size[0])

    images = Input(shape = (L, L, 3))
    x = Conv2D(64, (5, 5), strides = (2, 2), padding = 'same')(images)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(128, (5, 5), strides = (2, 2), padding = 'same')(x)
    x = LeakyReLU(0.2)(x)
    x = Flatten()(x)
    x = Dense(256)(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.5)(x)
    x = Dense(1)(x)
    outputs = Activation('tanh')(x)

    model = Model(inputs = images, outputs = outputs)

    model.summary()

    return model

# wasserstein : WGAN objective
# critic MAXIMIZE (f(x) - f(g(z)))/N
# -> minimize -(f(x) - f(g(z)))/N
# generator MINIMIZE (f(x) - f(g(z)))/N
# -> minimize -f(g(z))/N
def wasserstein(y_true, y_pred): # judged as y = 1:true, -1:fake

    return -K.mean(y_true * y_pred)
    


ResultPath = {}
ResultPath['image'] = 'image/'
ResultPath['model'] = 'model/'
for path in ResultPath:
    if not os.path.exists(path):
        os.mkdir(path)

        
def train(x_train, loadweight = False,
          lr_c = 2e-5, lr_g = 1e-5,
          BatchSize = 40, NumEpoch = 300):

    print('train data shape{}'.format(x_train.shape))
    image_num, image_size, _, _ = x_train.shape
    image_size = (image_size, image_size)
    
    # x_train.shape(samples, 64, 64, 3), unnormalized
    x_train = (x_train.astype(np.float32) - 127.5)/127.5

    c_model = CriticModel(image_size = image_size)
    c_opt = RMSprop(lr = lr_c)
    c_model.compile(loss = wasserstein, optimizer = c_opt)
    
    c_model.trainable = False
    g_model = GeneratorModel(image_size = image_size)
    wgan = Sequential([g_model, c_model])
    g_opt = RMSprop(lr = lr_g)
    wgan.compile(loss = wasserstein, optimizer = g_opt)

    if not loadweight == False:
        c_model.load_weights(filepath = loadweight+'/wgan_c.h5',
                             by_name = False)
        g_model.load_weights(filepath = loadweight+'/wgan_g.h5',
                             by_name = False)

    num_batches = int(x_train.shape[0]/BatchSize)
    print('Number of Batches : {}, epochs : {}'.format(num_batches, NumEpoch))

    for epoch in range(NumEpoch):

        if epoch > 0 and epoch%50 == 0:
            schedule = epoch%50

            c_opt = Adam(lr = lr_c/(10**schedule))
            c_model = compile(loss = wasserstein, optimizer = c_opt)

            c_model.trainable = False
            g_opt = Adam(lr = lr_g/(10**schedule))
            wgan = Sequential([g_model, c_model])
            wgan.compile(loss = wasserstein, optimizer = g_opt)

        for index in range(num_batches):

            # train critic(discriminator)
            c_train_num = 1
            for i in range(c_train_num):

                # weight clipping
                for l in c_model.layers:
                    weights = l.get_weights()
                    weigths = [np.clip(w, -0.01, 0.01) for w in weights]
                    l.set_weights(weights)

                # feed True/Fake image separately
                # train by true image
                idx = np.random.choice(image_num,
                                       BatchSize,
                                       replace = False)
                image_batch = x_train[idx]
                y = [1]*BatchSize
                c_loss_true = c_model.train_on_batch(image_batch, y)
                
                # train by generate fake image
                noise = np.array([np.random.uniform(-1, 1, 100)\
                                  for _ in range(BatchSize)])
                generated_images = g_model.predict(noise, verbose = 0)
                y = [-1]*BatchSize
                c_loss_fake = c_model.train_on_batch(generated_images, y)

                c_loss = (c_loss_true+c_loss_fake)/2

            # train generator
            noise = np.array([np.random.uniform(-1, 1, 100)\
                              for _ in range(BatchSize)])
            y = np.array([1]*BatchSize)
            g_loss = wgan.train_on_batch(noise, y)

            if index == num_batches-1:
                image = combine_images(generated_images)
                image = image*127.5 + 127.5

                Image.fromarray(image.astype(np.uint8))\
                    .save(ResultPath['image'] + '{}.png'.format(epoch))

            print('epoch:{}, batch:{}, g_loss:{}, c_loss:{}'.format(epoch,
                                                                    index,
                                                                    g_loss,
                                                                    c_loss))

    g_model.save_weights(ResultPath['model']+'wgan_g.h5')
    c_model.save_weights(ResultPath['model']+'wgan_c.h5')
