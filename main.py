# coding:utf-8

import os, sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from PIL import Image
import h5py
import argparse
import fire

from keras.models import Sequential
from keras.optimizers import RMSprop
import keras.backend as K

from model import generator, discriminator
from misc.utils import get_image

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
# data {/O
parser.add_argument('--image_target', type = int, default = 108,
                    help = 'target area of training data [108]')
parser.add_argument('--image_size', type = int, default = 64,
                    help = 'size of generated image [64]')
parser.add_argument('--datadir', type = str, nargs = '+', required = True,
                    help = 'path to directory contains training (image) data')
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
      'target size : {}, image size : {}\n'.format(args.image_target, args.image_size),
      'data dir : {}\n,'.format(args.datadir),
      'weight flag : {}, weight dir : {}, sample dir : {}'\
      .format(args.loadweight, args.weightdir, args.sampledir))



# wasserstein : WGAN objective
# critic MAXIMIZE (f(x) - f(g(z)))/N
# -> minimize -(f(x) - f(g(z)))/N
# generator MINIMIZE (f(x) - f(g(z)))/N
# -> minimize -f(g(z))/N
def wasserstein(y_true, y_pred): # y = 1:true, -1:fake

    return -K.mean(y_true * y_pred)
        
def train():

    # model construction
    disc = discriminator(image_size = args.image_size)
    d_opt = RMSprop(lr = args.lr_d)
    disc.compile(loss = wasserstein, optimizer = d_opt)

    disc.trainable = False
    gen = generator(image_size = args.image_size)
    wgan = Sequential([gen, disc])
    g_opt = RMSprop(lr = args.lr_g)
    wgan.compile(loss = wasserstein, optimizer = g_opt)

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
    # print(np.random.choice(paths, 10))
        
    epochs = args.epochs
    batch_size = args.batch_size
    num_batches = int(len(paths)/batch_size)
    print('Number of batches : {}, epochs : {}'.format(num_batches, epochs))

    for epoch in range(epochs):
        
        for batch in range(num_batches):
            
            disc.trainable = True
            for _ in range(args.nd):
                d_weights = [np.clip(w, -0.01, 0.01) for w in disc.get_weights]
                disc.set_weights(d_weights)

                # train by true images
                files = np.random.choice(paths, batch_size, replace = False)
                x_true = np.array([get_image(f, args.image_target, args.image_size)
                                   for f in files]) # # true images
                z = np.random.uniform(-1, 1, (batch_size, 100))
                x_fake = gen.predict(z) # fake images
                x = np.concatenate((x_true, x_fake))
                y = [1]*batch_size + [-1]*batch_size

                d_loss = disc.train_on_batch(x, y)

            # train generator
            disc.trainable = False
            z = np.random.uniform(-1, 1, (batch_size, 100))
            y = [1]*batch_size
            g_loss = wgan.train_on_batch(z, y)

            print('epoch:{}, batch:{}, g_loss:{}, d_loss:{}'\
                  .format(epoch, batch, g_loss, d_loss))

            checkpoint = np.array([int(num_batches*(i/20)) for i in range(20)])
            if batch in checkpoint:
                sample = combine_images(x_fake)
                sample = sample*127.5 + 127.5

                Image.fromarray(image.astype(np.uint8))\
                    .save(args.sampledir + '/sample{}_{}.png'.format(epoch, batch))

        gen.save_weights(args.weightdir + '/wgan_g.h5')
        disc.save_weights(args.weightdir + '/wgan_d.h5')

if __name__ == '__main__':

    train()
