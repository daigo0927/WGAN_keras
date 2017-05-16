# coding:utf-8

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

from keras.datasets import cifar10

def cifar10_extract(label = 'cat'):
    # acceptable label
    labels = ['airplane', 'automobile', 'bird', 'cat', 'deer',
              'dog', 'frog', 'horse', 'ship', 'truck']

    target_label = labels.index(label)

    (x_train, t_train), (x_test, t_test) = cifar10.load_data()

    t_target = t_train==target_label
    t_target = t_target.reshape(t_target.size)

    x_target = x_train[t_target]
    
    print('extract {} labeled images, shape(5000, 32, 32, 3)'.format(label))
    return x_target


# shape(generated_images) : (sample_num, w, h, 3)
def combine_images(generated_images):

    total, width, height, ch = generated_images.shape
    cols = int(math.sqrt(total))
    rows = math.ceil(float(total)/cols)

    combined_image = np.zeros((height*rows, width*cols, 3),
                              dtype = generated_images.dtype)

    for index, image in enumerate(generated_images):
        i = int(index/cols)
        j = index % cols
        combine_image[width*i:width*(i+1), height*j*height*(j+1), :]\
            = image

    return combine_image
