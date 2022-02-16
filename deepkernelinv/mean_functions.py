from gpflow.mean_functions import Constant
import tensorflow as tf


class ZeroImage(Constant):
    def __init__(self, output_dim=1):
        Constant.__init__(self)
        self.output_dim = output_dim
        del self.c

    def __call__(self, X):
        return tf.zeros((tf.shape(X)[0], self.output_dim), dtype=X.dtype)