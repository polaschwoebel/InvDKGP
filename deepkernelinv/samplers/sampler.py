import tensorflow as tf

from ..kernels import DeepKernel
from ..inducing_variables import KernelSpaceInducingPoints
from invgp.samplers import sample_matheron


@sample_matheron.register(object, KernelSpaceInducingPoints, DeepKernel, object, object)
def _sample_matheron(Xnew, inducing_variable, kernel, q_mu, q_sqrt, white=True, num_samples=1, num_basis=512):
    Xnew_transf = kernel.cnn(tf.reshape(Xnew, (-1,) + kernel.image_shape), training=False)
    #print(kernel.image_shape)
   # samples = sample_matheron(Xnew_transf, inducing_variable, kernel.basekern, q_mu, q_sqrt, white=white, num_samples=num_samples, num_basis=num_basis)
    #print(Xnew.shape)
    #print(Xnew_transf.shape)
    samples = sample_matheron(Xnew_transf, inducing_variable, kernel.basekern, q_mu, q_sqrt, white=white, num_samples=num_samples, num_basis=num_basis)
    return samples
