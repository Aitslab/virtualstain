import argparse

import os
import os.path
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import skimage.io
import skimage.morphology

import tensorflow as tf
import keras
import math

def predict(model, images, dim1, dim2, num_outputs=1):
    """Predict a multiple stack of 7 images to 1 channel

    Args:
        stack: ndarray of shape (height, width, 7)
    """
    xn = math.ceil(images.shape[1] / dim1)
    yn = math.ceil(images.shape[0] / dim2)
    
    output = np.zeros_like(images, shape=(images.shape[0], images.shape[1], num_outputs))

    # Create patches
    stacks = []
    for y in range(0, yn*dim2, dim2-32):
        for x in range(0, xn*dim1, dim1-32):
            left, top, right, bottom = x, y, x+dim1, y+dim2
            
            if right > images.shape[1]:
                left = images.shape[1]-dim1
                right = images.shape[1]

            if bottom > images.shape[0]:
                top = images.shape[0]-dim2
                bottom = images.shape[0]

            stacks.append(images[top:bottom,left:right])

    # Predicting
    stack = np.stack(stacks, axis=0)
    predictions = model.predict(stack, batch_size=16)

    # Stich (overlapping)
    print(predictions.shape)
    k = 0
    for y in range(0, yn*dim2, dim2-32):
        for x in range(0, xn*dim1, dim1-32):
            left, top, right, bottom = x, y, x+dim1, y+dim2
            relLeft, relTop, relRight, relBottom = 0, 0, dim1, dim2

            if right > images.shape[1]:
                left = images.shape[1]-dim1
                right = images.shape[1]

            if bottom > images.shape[0]:
                top = images.shape[0]-dim2
                bottom = images.shape[0]

            if top != 0:
                top += 16
                relTop += 16

            if left != 0:
                left += 16
                relLeft += 16

            if right != images.shape[1]:
                right -=16
                relRight -=16

            if bottom != images.shape[0]:
                bottom -=16
                relBottom -=16

            patch = predictions[k,:,:,:]
            
            output[top:bottom,left:right,:] = patch[relTop:relBottom,relLeft:relRight]
            k += 1

    return output