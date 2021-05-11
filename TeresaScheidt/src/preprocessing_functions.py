import skimage.io
import cv2
import numpy as np


def stack_images(images, num_channels):

    stack = []
    if num_channels == 1: 
        for image in images: 
            im = cv2.imread(image, cv2.IMREAD_UNCHANGED)
            if np.size(im.shape) > 2:
                im = im[:,:,0]
            stack.append(im)
        stack = np.stack(stack, axis=0)
    if num_channels == 2:
        num_outputs = int(len(images)/num_channels)
        for num in range(num_outputs):
            im1 = cv2.imread(images[2*num], cv2.IMREAD_UNCHANGED)
            im2 = cv2.imread(images[2*num+1], cv2.IMREAD_UNCHANGED)
            if np.size(im1.shape) > 2:
                im1 = im1[:,:,0]
                im2 = im2[:,:,0]
            im = np.stack([im1, im2], axis = 2)
            stack.append(im)
        stack = np.stack(stack, axis=0)
    
    return stack

def norm_batch(imagestack):
    mean = np.mean(imagestack)
    std = np.std(imagestack)
    norm_im = (imagestack - mean)/std 
    
    return norm_im, mean, std
    

def norm(imagestack, mean, std):
    """Normalization of imagestack
    
    this function takes a stack of images
    the array is normalized by the defined mean and std
    
    returns normalized imagestack 
    """
    
    new_im = (imagestack - mean)/std 
    
    return new_im

def recenter(imagestack, center=0.5, std=0.125):
    new_im = imagestack * std + center
    
    return new_im
 
def center_back(imagestack, center=0.5, std=0.125):
    new_im = (imagestack - center)/std
    
    return new_im
 
def unnormalize(images, mean, std):
    """Unnormalizes images 
    
    this function takes normalized images (with mean/std norm) and reverses the
    normalizaiton
    
    takes fixed mean and std from normalization 
    takes single images or stacks of images as arrays
    
    returns images """
    
    unnorm_images = images * std + mean
    
    
    return unnorm_images


    

def norm_and_stack(images):
    """Batch-Normalization and stacking of images
    
    this function takes a folder of images (e.g. defined by glob.glob(r"path\*.png"))
    and stacks the images into one array
    the array is normalized by the mean and std of whole array
    
    returns normalized imagestack, mean and std 
    
    """
    imagestack = np.dstack(tuple([cv2.imread(image, cv2.IMREAD_UNCHANGED) for image in images]))
    mean = np.mean(imagestack)
    std = np.std(imagestack)
    new_im = (imagestack - mean)/std 
    
    return new_im, mean, std
    


