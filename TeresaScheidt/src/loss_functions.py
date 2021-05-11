import tensorflow as tf

def SSIMLoss(y_true, y_pred):

    loss = 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0)).numpy()
    return loss
    
def mse(y_true, y_pred):

    loss = tf.keras.losses.MSE(y_true, y_pred)
    return loss
    
def mae(y_true, y_pred):
    
    loss = tf.keras.losses.MAE(y_true, y_pred)
    return loss