from skimage.metrics import structural_similarity as ssim
import numpy as np
   
def eval(model, x_images, y_images, dim):
    """ Evaluation of existing models

    takes a trained model with inout size [None,dim,dim], 
    a stack of imput-images of size [n,dim,dim] and a stack of target images of size [n,dim,dim,2]

    returns mean absolut error, mean squared error and structural similarity for each
    channel of the predicted images

    """

    pred_images = model.predict(x_images, batch_size = 8, verbose = 1)
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