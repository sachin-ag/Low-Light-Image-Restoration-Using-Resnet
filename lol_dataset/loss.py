import tensorflow as tf
from preprocess import get_color_map

def loss(y_true, y_pred):
    alpha = .5
    h = alpha * tf.keras.losses.Huber()(y_true, y_pred)
    true_color_map = get_color_map(y_true)
    pred_color_map = get_color_map(y_pred)
    h += (1 - alpha) * tf.keras.losses.Huber()(true_color_map, pred_color_map)
    return h

def psnr(y_true, y_pred):
    return tf.image.psnr(y_pred, y_true, max_val=1.0)