import sys
sys.path.append('../')
sys.path.append('../model')
import tensorflow as tf
import numpy as np

from tf_data_importer import load_training_tf_dataset
from custom_loss_functions import mse_ssim_loss

from hformer_model_extended import get_hformer_model

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

from tensorflow.keras.callbacks import ModelCheckpoint

import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from keras import backend as K

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

def split_dataset(dataset, split_ratio=0.8):
    total_samples = len(dataset)
    train_size = int(total_samples * split_ratio)

    train_dataset = dataset.take(train_size)
    val_dataset = dataset.skip(train_size)

    return train_dataset, val_dataset

# Define the custom PSNR metric
def psnr(y_true, y_pred):
    # Ensure the images have the same number of channels
    y_true = tf.convert_to_tensor(y_true)
    y_pred = tf.convert_to_tensor(y_pred)

    # Calculate the MSE
    mse = K.mean(K.square(y_true - y_pred))

    # Calculate the PSNR
    max_pixel = 1.0  # Assuming pixel values are normalized between 0 and 1
    psnr_value = 10.0 * K.log((max_pixel ** 2) / mse) / tf.math.log(10.0)

    return psnr_value 

def train_model(training_dataset, epochs, trained_model_file_name, history_file_name):        
    # Testing if model can be compiled
    # From the paper,
    # The batch size is 16 through 4000 epochs. 
    # The ADAM-W optimizer was used to minimize the mean squared error loss, and the learning rate was 1.0 × 10−5
    # AdamW cannot be used with tf2.10, so revering to Adam.

    model = get_hformer_model(num_channels_to_be_generated=4, name="hformer_model_64_channel")

    # Train validation split.
    train_dataset, val_dataset = split_dataset(training_dataset, split_ratio=0.8)

    model.build(input_shape=(None, 64, 64, 1)) 

    model.compile(tf.keras.optimizers.Adam(learning_rate=1.0 * 10**-5), metrics=[psnr, 'accuracy'], loss=mse_ssim_loss)
    print(model.summary())
    
    # Saving the model weights after every epoch.
    check_point_filepath="full_dataset_saved_weights/hformer_64_channel_custom_loss_epochs_{epoch:02d}.h5"
    checkpoint = ModelCheckpoint(check_point_filepath, monitor='val_accuracy', verbose=1, save_best_only=False, save_weights_only=True)
    
    history = model.fit(train_dataset, epochs=epochs,  verbose=1, validation_data=val_dataset, callbacks=[checkpoint])
    
    # Save the model weights to an HDF5 file
    # We cant use model.save as subclassing API is used here.
    model.save_weights(trained_model_file_name)
    
    # Save the training history
    np.save(history_file_name, history.history)
    
def main():
    training_dataset = load_training_tf_dataset(low_dose_ct_training_dataset_dir='../../../../../../Dataset/LowDoseCTGrandChallenge/Training_Image_Data', load_as_patches=True, load_limited_images=True, num_images_to_load=2)

    trained_model_file_name = 'hformer_200_epoch_extended_64_channel_custom_loss.h5'
    history_file_name = 'hformer_200_epoch_history_extended_64_channel_custom_loss.npy'
    
    print('training dataset' , training_dataset)
    
    train_model(training_dataset, 200, trained_model_file_name, history_file_name)
    
    print('model trained successfully with name : ', trained_model_file_name)
    print('saved history in file with name : ', history_file_name)
    
if __name__ ==  "__main__":
    main()
