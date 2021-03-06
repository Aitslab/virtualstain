import os
import os.path
import numpy as np
import keras.backend
import keras.layers
import keras.models

import tensorflow as tf
import skimage.io
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import loss_functions as lf


config_vars = {}

config_vars["learning_rate"] = 1e-4

config_vars["epochs"] = 50

config_vars["steps_per_epoch"] = 10

config_vars["pixel_depth"] = 16

config_vars["batch_size"] = 32

config_vars["val_batch_size"] = 32

config_vars["rescale_labels"] = True

config_vars["crop_size"] = 128

def random_sample_generator(x_images, y_images, batch_size, dim1, dim2, y_channels):
    do_augmentation = True
    while(True):    
        x_train = np.zeros((batch_size, dim1, dim2, 1), dtype=np.float32)
        y_train = np.zeros((batch_size, dim1, dim2, y_channels), dtype=np.float32) 
        max_index = x_images.shape[0] 
        # get one image at a time
        for i in range(batch_size):
                       
            # get random image
            img_index = np.random.randint(low=0, high=max_index)
            
            x = x_images[img_index]
            y = y_images[img_index]

            # get random crop
            start_dim1 = np.random.randint(low=0, high=x.shape[0] - dim1)
            start_dim2 = np.random.randint(low=0, high=x.shape[1] - dim2)

            patch_x = x[start_dim1:start_dim1 + dim1, start_dim2:start_dim2 + dim2] #* rescale_factor
            patch_y = y[start_dim1:start_dim1 + dim1, start_dim2:start_dim2 + dim2] #* rescale_factor_labels
            
            if(do_augmentation):
                
                rand_flip = np.random.randint(low=0, high=2)
                rand_rotate = np.random.randint(low=0, high=4)
                
                # flip
                if(rand_flip == 0):
                    patch_x = np.flip(patch_x, 0)
                    patch_y = np.flip(patch_y, 0)
                
                # rotate
                for rotate_index in range(rand_rotate):
                    patch_x = np.rot90(patch_x)
                    patch_y = np.rot90(patch_y)

                # illumination
                #ifactor = 1 + np.random.uniform(-0.25, 0.25) # Was before -0.75, 0.75
                #patch_x *= ifactor
                    
            # save image to buffer
            x_train[i, :, :, 0] = patch_x
            y_train[i, :, :, 0:y_channels] = patch_y
            
        # return the buffer
        yield(x_train, y_train)
    
        
CONST_DO_RATE = 0.5

option_dict_conv = {"activation": "relu", "padding": "same"}
option_dict_bn = {"momentum" : 0.9}
keras.backend.set_floatx("float32")

# returns a model from gray input to 64 channels of the same size
def get_model(dim1, dim2, y_channels):
    
    x = keras.layers.Input(shape=(dim1, dim2, 1), dtype="float32")
    #print('Input', x.shape)
    a = keras.layers.Convolution2D(64, 3, **option_dict_conv)(x)  
    a = keras.layers.BatchNormalization(**option_dict_bn)(a)
    
    a = keras.layers.Convolution2D(64, 3, **option_dict_conv)(a)
    a = keras.layers.BatchNormalization(**option_dict_bn)(a)

    
    y = keras.layers.MaxPooling2D()(a)
    #print('1. Conv', y.shape)
    b = keras.layers.Convolution2D(128, 3, **option_dict_conv)(y)
    b = keras.layers.BatchNormalization(**option_dict_bn)(b)

    b = keras.layers.Convolution2D(128, 3, **option_dict_conv)(b)
    b = keras.layers.BatchNormalization(**option_dict_bn)(b)
    #print('2. conv', b.shape)
    
    y = keras.layers.MaxPooling2D()(b)
    
    c = keras.layers.Convolution2D(256, 3, **option_dict_conv)(y)
    c = keras.layers.BatchNormalization(**option_dict_bn)(c)

    c = keras.layers.Convolution2D(256, 3, **option_dict_conv)(c)
    c = keras.layers.BatchNormalization(**option_dict_bn)(c)
    #print('3. conv', c.shape)
    
    y = keras.layers.MaxPooling2D()(c)
    
    d = keras.layers.Convolution2D(512, 3, **option_dict_conv)(y)
    d = keras.layers.BatchNormalization(**option_dict_bn)(d)

    d = keras.layers.Convolution2D(512, 3, **option_dict_conv)(d)
    d = keras.layers.BatchNormalization(**option_dict_bn)(d)
    #print('4. conv', d.shape)
    
    # UP

    d = keras.layers.UpSampling2D()(d)
    
    y = keras.layers.merge.concatenate([d, c], axis=3)
    #print('1. UpConv', y.shape)
    e = keras.layers.Convolution2D(256, 3,  **option_dict_conv)(y)
    e = keras.layers.BatchNormalization(**option_dict_bn)(e)

    e = keras.layers.Convolution2D(256, 3, **option_dict_conv)(e)
    e = keras.layers.BatchNormalization(**option_dict_bn)(e)

    e = keras.layers.UpSampling2D()(e)

    
    y = keras.layers.merge.concatenate([e, b], axis=3)
    #print('2. UpConv',y.shape)
    f = keras.layers.Convolution2D(128, 3, **option_dict_conv)(y)
    f = keras.layers.BatchNormalization(**option_dict_bn)(f)

    f = keras.layers.Convolution2D(128, 3, **option_dict_conv)(f)
    f = keras.layers.BatchNormalization(**option_dict_bn)(f)

    f = keras.layers.UpSampling2D()(f)

    
    y = keras.layers.merge.concatenate([f, a], axis=3)
    #print('3. UpConv',y.shape)
    y = keras.layers.Convolution2D(64, 3, **option_dict_conv)(y)
    y = keras.layers.BatchNormalization(**option_dict_bn)(y)

    y = keras.layers.Convolution2D(64, 3, **option_dict_conv)(y)
    y = keras.layers.BatchNormalization(**option_dict_bn)(y)
    y = keras.layers.Convolution2D(y_channels, 1, **option_dict_conv)(y)
    #y = keras.layers.Activation(activation='relu')(y) 

    model = keras.models.Model(x, y)
    return model
    
    
def train(train_images_x,train_images_y, validation_images_x, validation_images_y, y_channels, name, model=None):

    # build session running on GPU 1
    configuration = tf.compat.v1.ConfigProto()
    configuration.gpu_options.allow_growth = True
    configuration.gpu_options.visible_device_list = "0"
    session = tf.compat.v1.Session(config = configuration)

    # apply session
    tf.compat.v1.keras.backend.set_session(session)


    train_gen = random_sample_generator(
        train_images_x, train_images_y,
        config_vars["batch_size"],
        config_vars["crop_size"],
        config_vars["crop_size"],
        y_channels
    )

    val_gen = random_sample_generator(
        validation_images_x, validation_images_y,
        config_vars["val_batch_size"],
        config_vars["crop_size"],
        config_vars["crop_size"],
        y_channels
    )
    
    

    # build model
    if model == None: 
        model = get_model(config_vars["crop_size"], config_vars["crop_size"], y_channels)
    else:
        model = model
    
    model.summary()
    
    SSIM = lf.SSIMLoss
    
    mse_ssim = lf.mse_ssim
    
    mse = tf.keras.losses.MSE
    
    loss = mse_ssim

    metrics = ["mse", "mae"]

    optimizer = keras.optimizers.Adam(lr=config_vars["learning_rate"], clipvalue=0.1)

    model.compile(loss=loss, metrics=metrics, optimizer=optimizer)

    callback_csv = keras.callbacks.CSVLogger(filename=r"C:\Users\tsche\Desktop\Master\2. Semester\Project Staining\gitrepo\TeresaScheidt\models\checkpoints\tmp")

    checkpoint = ModelCheckpoint(name, monitor='val_loss', verbose=1,
        save_best_only=True, mode='auto')
    
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=1,
        mode='auto', baseline=None, restore_best_weights=False)

    # TRAIN
    try:
        statistics = model.fit(
            train_gen,
            steps_per_epoch=config_vars["steps_per_epoch"],
            epochs=config_vars["epochs"],
            validation_data=val_gen,
            validation_steps=10,
            callbacks = [checkpoint, early_stop], #, TensorboardBatch(8, log_dir=paths["log_dir"])],
            verbose = 1
        )
    except KeyboardInterrupt as e:
        print("Aborted...")

    #print("Saving model...")
    #model.save(name)
    print('Done! :)')
    return statistics
    
    
    
def eval(model, x_images, y_images):

    image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)

# Add dimension to match with input mode 
    image_tensor = tf.expand_dims(image_tensor, 0)
    
    image_gen = random_sample_generator(
    x_images, y_images,
    config_vars["batch_size"],
    config_vars["crop_size"],
    config_vars["crop_size"],
    2)
    
    metrics = model.evaluate(image_gen, batch_size=32, steps=10)
    
    return metrics