# coding:utf-8

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math


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
