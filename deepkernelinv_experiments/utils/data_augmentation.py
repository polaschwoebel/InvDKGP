import tensorflow as tf
import numpy as np
from tensorflow.keras import layers


def augment_color(data, orbit):
    image = data[0]
    label = data[1]
    batch_size = image.shape[0]
    augmented = orbit(image)
    idx = tf.constant(list(zip(range(batch_size), range(batch_size))), tf.int32)
    image = tf.gather_nd(augmented, idx)
    return image, label


def augment_affine(data, orbit):
    image = data[0]
    label = data[1]
    batch_size = image.shape[0]
    augmented = orbit(image)
    idx = tf.constant(list(zip(range(batch_size), range(batch_size))), tf.int32)
    image = tf.gather_nd(augmented, idx)
    return image, label


def augment_rotate_orbit(data, orbit):
    image = data[0]
    label = data[1]
    batch_size = image.shape[0]
    augmented = orbit(image)
    image = tf.squeeze(augmented, 1)
    #idx = tf.constant(list(zip(range(batch_size), range(batch_size))), tf.int32)
    #image = tf.gather_nd(augmented, idx)
    return image, label


def augment_rot_quant_orbit(data, orbit):
    image = data[0]
    label = data[1]
    batch_size = image.shape[0]
    augmented = orbit(image)
    idx = tf.constant(list(zip(range(batch_size), list(range(2))*32)), tf.int32)
    image = tf.gather_nd(augmented, idx)
    return image, label


def random_rotate_img(x):
    # k = np.random.choice(3)
    k = tf.random.uniform([], 0, 5)
    k = tf.cast(k, tf.int32)
    # if tf.random.uniform([]) > 0.5:
    x = tf.image.rot90(x, k=k)
    return x


horizontal_flip = tf.keras.layers.experimental.preprocessing.RandomFlip(
    mode="horizontal")
horizontal_and_vertical_flip = tf.keras.layers.experimental.preprocessing.RandomFlip(
    mode="horizontal_and_vertical")
translate_zeropad = tf.keras.layers.experimental.preprocessing.RandomTranslation(
    height_factor=0.0625, width_factor=0.0625, fill_mode='constant')  # corresponding to 4 pixels for PCAM
translate_nearestpad = tf.keras.layers.experimental.preprocessing.RandomTranslation(
    height_factor=0.0625, width_factor=0.0625, fill_mode='nearest')


def augment_translate_zeropad(image, label):
    image = translate_zeropad(image)
    return image, label


def augment_translate_nearestpad(image, label):
    image = translate_nearestpad(image)
    return image, label


def augment_mirror_translate_zeropad(image, label):
    image = horizontal_and_vertical_flip(image)
    image = translate_zeropad(image)
    return image, label


def augment_mirror_translate_nearestpad(image, label):
    image = horizontal_and_vertical_flip(image)
    image = translate_nearestpad(image)
    return image, label


def augment_mirror(image, label):
    image = horizontal_and_vertical_flip(image)
    return image, label


def augment_rotate(image, label):
    image = random_rotate_img(image)
    return image, label