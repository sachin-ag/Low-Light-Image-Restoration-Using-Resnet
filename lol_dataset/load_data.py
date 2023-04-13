import random
import tensorflow as tf
from glob import glob
from preprocess import *

random.seed(10)

IMAGE_SIZE = 128
BATCH_SIZE = 4

def convert_to_12_channels(low_image):
    low_image.set_shape([None, None, 3])
    low_image = tf.image.adjust_gamma(low_image)
    eq_image = histogram_equalisation(low_image)
    eq_image.set_shape([None, None, 3])
    color_map = get_color_map(low_image)
    color_map.set_shape([None, None, 3])
    noise_map = get_noise_map(color_map)
    noise_map.set_shape([None, None, 3])
    color_map = tf.cast(color_map, tf.experimental.numpy.uint8)
    noise_map = tf.cast(noise_map, tf.experimental.numpy.uint8)
    merged_image = tf.concat([low_image, eq_image, color_map, noise_map], axis=-1)
    merged_image.set_shape([None, None, 12])
    merged_image = tf.cast(merged_image, dtype=tf.float32) / 255.0
    return merged_image

def read_image_low(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image = convert_to_12_channels(image)
    return image

def read_image_enhanced(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image.set_shape([None, None, 3])
    image = tf.cast(image, dtype=tf.float32) / 255.0
    return image

def load_data(low_light_image_path, enhanced_image_path):
    low_light_image = read_image_low(low_light_image_path)
    enhanced_image = read_image_enhanced(enhanced_image_path)
    low_light_image, enhanced_image = random_crop(low_light_image, enhanced_image, IMAGE_SIZE)
    return low_light_image, enhanced_image

def get_dataset(low_light_images, enhanced_images):
    dataset = tf.data.Dataset.from_tensor_slices((low_light_images, enhanced_images))
    dataset = dataset.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    return dataset

def get_train_dataset():
    train_low_light_images = sorted(glob("./lol_dataset/our485/low/*"))
    train_enhanced_images = sorted(glob("./lol_dataset/our485/high/*"))
    train_dataset = get_dataset(train_low_light_images, train_enhanced_images)
    return train_dataset

def get_val_dataset():
    val_low_light_images = sorted(glob("./lol_dataset/eval15/low/*"))
    val_enhanced_images = sorted(glob("./lol_dataset/eval15/high/*"))
    val_dataset = get_dataset(val_low_light_images, val_enhanced_images)
    return val_dataset