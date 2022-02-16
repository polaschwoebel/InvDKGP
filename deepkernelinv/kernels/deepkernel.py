import tensorflow as tf
from typing import Optional, Tuple
import gpflow


class DeepKernel(gpflow.kernels.Kernel):
    def __init__(self, cnn, image_shape: Tuple,
                 basekern: gpflow.kernels.Kernel):
        super().__init__()
        with self.name_scope:  # what is this line and is it necessary?
            self.basekern = basekern  # base kernel is the kernel then applied to the mapped inputs
            self.image_shape = image_shape
            # input_size = int(tf.reduce_prod(image_shape))
            # input_shape = (input_size, )\
            self.cnn = cnn
            self.cnn.build(image_shape)  # initialize weights

    def K(self, a_input: tf.Tensor, b_input: Optional[tf.Tensor] = None) -> tf.Tensor:
        transformed_a = self.cnn(tf.reshape(a_input, (-1,) + self.image_shape), training=False) 
        transformed_b = self.cnn(tf.reshape(b_input, (-1,) + self.image_shape), training=False) if b_input is not None else b_input
        return self.basekern.K(transformed_a, transformed_b)

    def K_diag(self, a_input: tf.Tensor) -> tf.Tensor:  # just a more efficient version to compute K(x, x)
        transformed_a = self.cnn(a_input, training=False)
        return self.basekern.K_diag(transformed_a)