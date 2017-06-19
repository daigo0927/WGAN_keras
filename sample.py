# coding:utf-8

import numpy as np
import matplotlib.pyplot
import seaborn as sns
import keras
from misc.utils import *
from PIL import Image
import h5py
from model import generator_deconv

def main(num_sample = 10, parallel = 16):
    gen = generator_deconv(64)
    gen.load_weights('./result/wgan_g_5epoch.h5')
    for i in range(num_sample):
        z = np.random.uniform(-1, 1, (parallel, 100))
        xfake = gen.predict(z)
        sample = sample*127.5 + 127.5

        Image.fromarray(sample.astype(np.uint8))\
            .save('./sample/image{}.png'.format(i))


if __name__ == '__main__':
    main()
    
