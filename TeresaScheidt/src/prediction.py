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

import loss_functions as lf

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
   
from skimage.metrics import structural_similarity as ssim
   
def eval(model, x_images, y_images, dim):
    pred_images = model.predict(x_images, batch_size = 4, verbose = 1)
    print('Image prediction done')
    
    MAE1 = np.mean(lf.mae(y_images[:,:,:,0], pred_images[:,:,:,0]))
    MSE1 = np.mean(lf.mse(y_images[:,:,:,0], pred_images[:,:,:,0]))
    MAE2 = np.mean(lf.mae(y_images[:,:,:,1], pred_images[:,:,:,1]))
    MSE2 = np.mean(lf.mse(y_images[:,:,:,1], pred_images[:,:,:,1]))
    SSIM1 = 0
    SSIM2 = 0
    for y_true, y_pred in zip(y_images[:,:,:,0], pred_images[:,:,:,0]):
        SSIM1 += ssim(y_true, y_pred, data_range=y_true.max() - y_true.min())
    SSIM1 /= y_images.shape[0]
    for y_true, y_pred in zip(y_images[:,:,:,1], pred_images[:,:,:,1]):
        SSIM2 += ssim(y_true, y_pred, data_range=y_true.max() - y_true.min())   
    SSIM2 /= y_images.shape[0]

    
    
    return MAE1, MSE1, SSIM1, MAE2, MSE2, SSIM2