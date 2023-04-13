import tensorflow as tf
from tensorflow import keras

def residual_block(input_tensor, channels):
    residue = tf.keras.layers.Conv2D(channels, (1,1))(input_tensor)
    conv1 = keras.layers.Conv2D(channels, kernel_size=(3, 3), padding="same", activation = 'relu')(input_tensor)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv2 = keras.layers.Conv2D(channels, kernel_size=(3, 3), padding="same")(conv1)
    conv2 = tf.keras.layers.Add()([conv2, residue]) 
    conv2 = tf.keras.layers.Activation('relu')(conv2)
    return conv2


def resnet(input_tensor):
    x = keras.layers.Conv2D(3, kernel_size=(3, 3), padding="same", activation = 'relu')(input_tensor)
    r1 = residual_block(x, 64)
    r2 = residual_block(r1, 128)
    r3 = residual_block(r2, 64)
    output_tensor = keras.layers.Conv2D(3, kernel_size=(3, 3), padding="same", activation = 'sigmoid')(r3)
    return output_tensor


def build_model():
    input_tensor = keras.Input(shape=[None, None, 12])
    orig_im, hist_eq, color_mp, noise_mp = tf.split(input_tensor, 4, axis=-1)

    x1 = resnet(orig_im)
    x1 = keras.layers.Add()([x1, orig_im])

    x2 = resnet(hist_eq)
    x2 = keras.layers.Add()([x2, hist_eq])

    x3 = resnet(color_mp)
    x3 = keras.layers.Add()([x3, color_mp])

    x4 = resnet(noise_mp)
    x4 = keras.layers.Add()([x4, noise_mp])

    output_tensor = tf.concat([x1, x2, x3, x4], axis=-1)
    output_tensor = keras.layers.Conv2D(3, kernel_size=(1, 1), padding="same", activation = 'sigmoid')(output_tensor)

    return keras.Model(input_tensor, output_tensor)