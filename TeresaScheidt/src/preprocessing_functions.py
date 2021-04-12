import skimage.io
import numpy as np


def normalize(input_path, output_path):
    """Normalization of image
    
    This function takes an image from <input_path> and normalizes 
    is according to (im - mean)/std and projects to [0,1]
    
    saves normalized image as 8bit image to <output_path>"""
    

    orig_im = skimage.io.imread(input_path)
    mean = np.mean(orig_im)
    std = np.std(orig_im)
    new_im = (orig_im - mean)/std 
    min_value = np.min(new_im)
    max_value = np.max(new_im)
    im = (new_im - min_value) / (max_value - min_value) 
    im = skimage.img_as_ubyte(im)
    skimage.io.imsave(output_path, im)
    

def norm_and_stack(images):
    """Batch-Normalization and stacking of images
    
    this function takes a folder of images (e.g. defined by glob.glob(r"path\*.png"))
    and stacks the images into one array
    the array is normalized by the mean and std of whole array
    
    returns normalizes imagestack, mean and std 
    
    """
    imagestack = np.dstack(tuple([skimage.io.imread(image) for image in images]))
    mean = np.mean(imagestack)
    std = np.std(imagestack)
    new_im = (imagestack - mean)/std 
    
    return new_im, mean, std

    
def unnormalize(images, mean, std):
    """Unnormalizes images 
    
    this function takes normalized images (with mean/std norm) and reverses the
    normalizaiton
    
    takes fixed mean and std from normalization 
    takes single images or stacks of images as arrays
    
    returns images as uint8"""
    
    unnorm_images = images * std + mean
    unnorm_images = unnorm_images.astype(np.uint8)
    
    return unnorm_images


    
    


