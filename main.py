# coding:utf-8

import os, sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import pdb
from PIL import Image
import h5py
import argparse
import fire

from keras.models import Sequential, Model
from keras.optimizers import RMSprop
from keras.layers import Input
import keras.backend as K

from model import *
from misc.utils import *

parser = argparse.ArgumentParser()
# optimization
parser.add_argument('-e', '--epochs', type=int, default=20,
                    help = 'number of epochs [20]')
parser.add_argument('--lr_g', type = float, default = 5e-5,
                    help = 'learning rate for generator [5e-5]')
parser.add_argument('--lr_d', type = float, default = 5e-5,
                    help = 'learning rate for discriminator [5e-5]')
parser.add_argument('--train_size', type = int, default = np.inf,
                    help = 'size of trainind data [np.inf]')
parser.add_argument('--batch_size', type = int, default = 64,
                    help = 'size of mini-batch [64]')
parser.add_argument('--nd', type = int, default = 5,
                    help = 'training schedule for dicriminator by generator [5]')
parser.add_argument('--generator', type = str, default = 'deconv',
                    choices = ['deconv', 'upsampling'],
                    help = 'choose generator type [deconv]')
# data {/O
parser.add_argument('--image_target', type = int, default = 108,
                    help = 'target area of training data [108]')
parser.add_argument('--image_size', type = int, default = 64,
                    help = 'size of generated image [64]')
parser.add_argument('--datadir', type = str, nargs = '+', required = True,
                    help = 'path to directory contains training (image) data')
parser.add_argument('--splitload', type = int, default = 5,
                    help = 'load data, by [5] split')
parser.add_argument('--loadweight', type = str, default = False,
                    help = 'path to directory conrtains trained weights [False]')
parser.add_argument('--weightdir', type = str, default = './model',
                    help = 'path to directory put trained weighted [./model]')
parser.add_argument('--sampledir', type = str, default = './image',
                    help = 'path to directory put generated image samples [./image]')
args = parser.parse_args()

print('epochs : {}, lr_g : {}, lr_d : {}\n'.format(args.epochs, args.lr_g, args.lr_d),
      'train size : {}, batch size : {}, disc-schedule : {}\n'\
      .format(args.train_size, args.batch_size, args.nd),
      'generator type : {},'.format(args.generator),
      'target size : {}, image size : {}\n'.format(args.image_target, args.image_size),
      'data dir : {}\n,'.format(args.datadir),
      'load data splitingly : {}\n'.format(args.splitload),
      'weight flag : {}, weight dir : {}, sample dir : {}'\
      .format(args.loadweight, args.weightdir, args.sampledir))


# wasserstein : WGAN objective
# critic MAXIMIZE (f(x) - f(g(z)))/N
# -> minimize -(f(x) - f(g(z)))/N
# generator MINIMIZE (f(x) - f(g(z)))/N
# -> minimize -f(g(z))/N
def wasserstein(y_true, y_pred): # y = 1:true, -1:fake

    return -K.mean(y_true * y_pred, axis = -1)
        
def train():

    # discriminator construction
    disc = discriminator(image_size = args.image_size)
    d_opt = RMSprop(lr = args.lr_d)
    disc.compile(loss = wasserstein, optimizer = d_opt)

    # Wasserstein GAN construction
    disc.trainable = False
    if args.generator == 'deconv':
        gen = generator_deconv(image_size = args.image_size)
    elif args.generator == 'upsampling':
        gen = generator_upsampling(image_size = args.image_size)
        print('warning! upsampling genearator usually fail to learn, sorry...')
    wgan = Sequential([gen, disc])
    g_opt = RMSprop(lr = args.lr_g)
    wgan.compile(loss = wasserstein, optimizer = g_opt)
    wgan.summary()

    # load weight (if needed)
    if not args.loadweight == False:
        disc.load_weights(filepath = args.loadweight+'/wgan_d.h5',
                          by_name = False)
        gen.load_weights(filepath = args.loadweight+'/wgan_g.h5',
                         by_name = False)

    # data(image) path list
    paths = []
    for ddir in args.datadir:
        paths = paths + glob.glob(ddir + '/*')
    datasize = min(len(paths), args.train_size)
    print('data size : {}'.format(datasize))
    paths = np.random.choice(paths, datasize, replace = False)

    # trainig schedule
    epochs = args.epochs
    batch_size = args.batch_size
    num_batches = int(len(paths)/batch_size)
    print('Number of batches : {}, epochs : {}'.format(num_batches, epochs))

    # training
    for epoch in range(epochs):
        
        for batch in range(num_batches):

            # load data splitingly
            if batch in np.linspace(0, num_batches, args.splitload+1, dtype = int):
                path_split = np.random.choice(paths,
                                              int(len(paths)/args.splitload),
                                              replace = False)
                data = np.array([get_image(p,
                                           args.image_target,
                                           args.image_size)\
                                 for p in path_split])


            # train discriminator
            # boost discriminator iteration
            # referred in https://github.com/martinarjovsky/WassersteinGAN
            if epoch == 0 and batch < 25:
                nd = 100
            elif batch%500 > 0 and batch%500 < 10:
                nd = 100
            else:
                nd = args.nd
            for _ in range(nd):
                d_weights = [np.clip(w, -0.01, 0.01) for w in disc.get_weights()]
                disc.set_weights(d_weights)
                # train by true image
                x_true = data[np.random.choice(len(data),
                                               batch_size,
                                               replace = False)]
                d_loss_true = disc.train_on_batch(x_true, [1]*batch_size)
                # train by fake image
                z = np.random.uniform(-1, 1, (batch_size, 100))
                x_fake = gen.predict(z)
                d_loss_fake = disc.train_on_batch(x_fake, [-1]*batch_size)

                d_loss = (d_loss_true + d_loss_fake)/2

            # train generator
            z = np.random.uniform(-1, 1, (batch_size, 100))
            y = [1]*batch_size
            g_loss = wgan.train_on_batch(z, y)

            print('epoch:{}, batch:{}, g_loss:{}, d_loss:{}'\
                  .format(epoch, batch, g_loss, d_loss))

            
            if batch%100 == 0:
                sample = combine_images(x_fake)
                sample = sample*127.5 + 127.5

                Image.fromarray(sample.astype(np.uint8))\
                     .save(args.sampledir + '/sample_{}_{}.png'.format(epoch, batch))

        gen.save_weights(args.weightdir + '/wgan_g_{}epoch.h5'.format(epoch))
        disc.save_weights(args.weightdir + '/wgan_d_{}epoch.h5'.format(epoch))

if __name__ == '__main__':

    train()
