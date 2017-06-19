# coding:utf-8

import numpy as np
import matplotlib.pyplot
import seaborn as sns
from keras.models import Model
from misc.utils import *
import pandas as pd
from PIL import Image
import h5py
from model import generator_deconv
import argparse
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--param',
                        type = str, default = './result/wgan_g_5epoch.h5',
                        help = 'trained parameter for generator [./result/wgan_g_5epoch.h5]')
    parser.add_argument('-ns', '--num_sample',
                        type = int, default = 10,
                        help = 'number of sampling images [10]')
    parser.add_argument('-np', '--num_parallel',
                        type = int, default = 9,
                        help = 'number of images put in the same [9]')
    parser.add_argument('-d', '--distdir',
                        type = str, default = './sample',
                        help = 'path to the directory images in [./sample]')
    args = parser.parse_args()

    sample(param = args.param,
           distdir = args.distdir,
           num_sample = args.num_sample,
           parallel = args.num_parallel)

def sample(param, distdir,
           num_sample = 10, parallel = 16):

    gen = generator_deconv(64)
    gen.load_weights(param)
    for i in tqdm(range(num_sample)):
        z = np.random.uniform(-1, 1, (parallel, 100))
        xfake = gen.predict(z)
        sample = combine_images(xfake)
        sample = sample*127.5 + 127.5

        Image.fromarray(sample.astype(np.uint8))\
            .save(distdir + '/image{}.png'.format(i))
        

if __name__ == '__main__':
    main()
    
    
