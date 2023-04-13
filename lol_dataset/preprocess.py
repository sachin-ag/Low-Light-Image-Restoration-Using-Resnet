import tensorflow as tf
import tensorflow_addons as tfa

def histogram_equalisation(image):
    eqalized_image = tfa.image.equalize(image)
    return eqalized_image

def get_color_map(img):
    mean_vals = tf.reduce_mean(img, axis=2, keepdims=True)
    color_img = img / mean_vals
    return color_img

def get_noise_map(color_map):
    color_map_expanded = tf.expand_dims(color_map, axis=0)
    gx, gy = tf.image.image_gradients(color_map_expanded)
    abs_gx = tf.abs(gx)
    abs_gy = tf.abs(gy)
    noise_map = tf.maximum(abs_gx, abs_gy)
    return tf.squeeze(noise_map)
  
def random_crop(low_image, enhanced_image, IMAGE_SIZE):
    low_image_shape = tf.shape(low_image)[:2]
    low_w = tf.random.uniform(shape=(), maxval=low_image_shape[1] - IMAGE_SIZE + 1, dtype=tf.int32)
    low_h = tf.random.uniform(shape=(), maxval=low_image_shape[0] - IMAGE_SIZE + 1, dtype=tf.int32)
    enhanced_w = low_w
    enhanced_h = low_h
    low_image_cropped = low_image[low_h : low_h + IMAGE_SIZE, low_w : low_w + IMAGE_SIZE]
    enhanced_image_cropped = enhanced_image[enhanced_h : enhanced_h + IMAGE_SIZE, enhanced_w : enhanced_w + IMAGE_SIZE]
    return low_image_cropped, enhanced_image_cropped
