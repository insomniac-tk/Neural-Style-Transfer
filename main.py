'''
@author : Tejas Khanna

'''
import argparse
import functools
import time

import os
import tensorflow as tf


import IPython.display as display

import matplotlib.pyplot as plt
import matplotlib as mpl

import numpy as np
import PIL.Image

import tensorflow_hub as hub
import utils

# Load compressed models from tensorflow_hub
os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'


# Set default fonts for matplotlib
font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 22}

mpl.rc('font', **font)

mpl.rcParams['figure.figsize'] = (12, 12)
mpl.rcParams['axes.grid'] = False


def stylize(content_image_path,
            style_image_path,
            model):
    content_image = utils.load_img(content_image_path)
    
    style_image = utils.load_img(style_image_path)
    result_image = model(tf.constant(content_image), tf.constant(style_image))[0]
    utils.plot_images(content_image, style_image, result_image)



if __name__ == "__main__":
    parser = argparse.ArgumentParser("Neural Style Transfer For Text")
    parser.add_argument("--style-image","-s",
                        help="Path to the style image.",
                        type=str,
                        required=True)
    parser.add_argument("--content-image","-c",
                        help="Path to the content image.",
                        type = str,
                        required=True)
    args = parser.parse_args()
    
    style_image = args.style_image
    content_image = args.content_image
    if not utils.validate_path(style_image) or not utils.validate_path(content_image):
        print("Invalid path for this image. {}".format(style_image))
        exit(1)
    try:
        model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
        stylize(content_image_path=content_image,
                style_image_path=style_image,
                model=model)
    except Exception as e:
        print("Exception {} has occured.".format(e))