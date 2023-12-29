
import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from keras import backend as K

# Define the custom PSNR loss function
def psnr(y_true, y_pred):
    # Ensure the images have the same number of channels
    y_true = tf.convert_to_tensor(y_true)
    y_pred = tf.convert_to_tensor(y_pred)

    # Calculate the MSE
    mse = K.mean(K.square(y_true - y_pred))

    # Calculate the PSNR
    max_pixel = 1.0  # Assuming pixel values are normalized between 0 and 1
    psnr_value = 10.0 * K.log((max_pixel ** 2) / mse) / tf.math.log(10.0)

    return -psnr_value  # Return the negative PSNR as a loss (to minimize)


# SSIM loss function
def ssim_loss(y_true, y_pred):
  return 1.0 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))

# MSE_SSIM loss function
def mse_ssim_loss(y_true, y_pred):
    mse_loss = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
    ssim_loss_value = ssim_loss(y_true, y_pred)

    # Define the weights for each loss function. It should technically sum to 1, but as weight is already low setting it to > 1.
    mse_weight = 1.0
    ssim_weight = 1.0

    combined_loss = mse_weight * mse_loss + ssim_weight * ssim_loss_value

    return combined_loss