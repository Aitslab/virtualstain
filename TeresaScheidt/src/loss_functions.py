import tensorflow as tf

def SSIMLoss(y_true, y_pred):

    loss = 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, tf.reduce_max(y_true) - tf.reduce_min(y_pred)))
    return loss
    
def mse(y_true, y_pred):

    loss = tf.keras.losses.MSE(y_true, y_pred)
    return loss
    
def mae(y_true, y_pred):
    
    loss = tf.keras.losses.MAE(y_true, y_pred)
    return loss
    
def mse_ssim(y_true, y_pred):
    
    loss = mse(y_true, y_pred) + SSIMLoss(y_true, y_pred)
    return loss