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
    

# def noise_reduction():
    


# def downsizing():
    
    


