import tensorflow as tf
from gpflow.config import default_float, default_int


def identity(*args):
    return args


def map_fn_flat_onehot(image, label, nr_classes):
    # GP needs flat shape
    label = tf.one_hot(tf.cast(label, default_int()), nr_classes) 
    label = tf.cast(label, default_float())
    image = tf.cast(image, default_float())
    image_shape = tf.math.reduce_prod(image.shape[1:])
    image = tf.reshape(image, [-1, image_shape])
    return image, label


def map_fn_flat_int(image, label, nr_classes=None):
    label = tf.expand_dims(label, 1)
    label = tf.cast(label, default_int())  # tf.float64)
    image = tf.cast(image, default_float())
    image_shape = tf.math.reduce_prod(image.shape[1:])
    image = tf.reshape(image, [-1, image_shape])
    return image, label


def map_fn_image_onehot(image, label, nr_classes):
    # CNN/DeepKernel need image shape
    image = tf.cast(image, default_float())
    onehot_label = tf.one_hot(tf.cast(label, tf.uint8), nr_classes)
    label = tf.cast(onehot_label, default_float())
    return image, label


def map_fn_image_int(image, label, nr_classes=None):
    image = tf.cast(image, default_float())
    label = tf.expand_dims(label, 1)
    label = tf.cast(label, default_int())  
    return image, label
