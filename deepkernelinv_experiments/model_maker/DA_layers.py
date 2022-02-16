import tensorflow as tf
from invgp.kernels.orbits import InterpretableSpatialTransform
import numpy as np
from invgp.kernels.image_transforms import rotate_img_angles, rotate_img_angles_stn, apply_stn_batch, _stn_theta_vec, \
    apply_stn_batch_colour


class DALayer(tf.keras.layers.Layer):
    def __init__(self, args):
        super(DALayer, self).__init__()
        theta_min_init = np.array([args.radian_init, 1., 1., 0., 0., 0, 0.]) - args.affine_init * np.array([3.14, 1., 1., 1., 1., 1., 1.]) , # play with initialization
        theta_max_init = np.array([-args.radian_init, 1., 1., 0., 0., 0., 0.]) + args.affine_init * np.array([3.14, 1., 1., 1., 1., 1., 1.]) ,  # play with initialization
        self.radians = args.radians
        self.batch_size = args.batch_size[0]
        self.image_shape = args.image_shape
        self.orbit_size = args.orbit_size[0]
        self.theta_min = tf.Variable(theta_min_init, dtype=tf.float32, trainable=True)
        self.theta_max = tf.Variable(theta_max_init, dtype=tf.float32, trainable=True)

    def orbit_minibatch(self, X):
        eps = tf.random.uniform([self.orbit_size, 7], 0., 1., dtype=tf.float32)
        # only unconstrained version for now
        theta_min = tf.reshape(self.theta_min, [1, -1])
        theta_max = tf.reshape(self.theta_max, [1, -1])
        thetas = theta_min + (theta_max - theta_min) * eps
        stn_thetas = tf.map_fn(lambda thetas: _stn_theta_vec(thetas, radians=self.radians), thetas)
        Ximgs = tf.reshape(X, [-1, self.image_shape[0], self.image_shape[0]])
        return apply_stn_batch(Ximgs, stn_thetas)

    def call(self, inputs):
        # print('input shape is', inputs.shape)
        orbit_samples = self.orbit_minibatch(inputs)
        # print('orbit samples shape is', orbit_samples)
        orbit_samples = tf.reshape(orbit_samples, [self.batch_size * self.orbit_size, self.image_shape[0], self.image_shape[1], 1])
        # print('orbit samples shape is', orbit_samples)
        return orbit_samples


class AggregatePredictionsLayer(tf.keras.layers.Layer):
    def __init__(self, args):
        super(AggregatePredictionsLayer, self).__init__()
        self.batch_size = args.batch_size[0]
        self.nr_classes = args.nr_classes
        self.orbit_size = args.orbit_size[0]

    def call(self, inputs):
        inputs = tf.reshape(inputs, [self.batch_size, self.orbit_size, self.nr_classes])
        aggregated = tf.math.reduce_mean(inputs, axis=1)
        return aggregated
