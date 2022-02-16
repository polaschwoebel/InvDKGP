import gpflow
import tensorflow as tf
from deepkernelinv.inducing_variables import KernelSpaceInducingPoints, ConvolvedKernelSpaceInducingPoints, StochasticConvolvedKernelSpaceInducingPoints
from invgp.inducing_variables.invariant_convolution_domain import ConvolvedInducingPoints, StochasticConvolvedInducingPoints
# from invgp.covariances import invariant_covariances

#from deepkernelinv.kernels import DeepKernel
from deepkernelinv.kernels import DeepKernel
from gpflow.base import TensorLike
from invgp.kernels import Invariant, StochasticInvariant
from gpflow.covariances.dispatch import Kuf, Kuu
import numpy as np 
# 0. for the shallow InvGP we can just use the (Stochastic)Inducing points from invgp directly

###############################################################################


# 1. DeepKernelGP: DeepKernel <- RBFKernel  ("A <- B" indicating B is A's basekern)
# u are in the kernel space, i.e. they have already been fed through the CNN
@Kuu.register(KernelSpaceInducingPoints, DeepKernel)
def Kuu_deep_kernel(inducing_variable, kernel, jitter=0.0):
    # print('CALLING DeepKernelGPs Kuu')
    Kzz = kernel.basekern.K(inducing_variable.Z)
    Kzz += jitter * tf.eye(inducing_variable.Z.shape[0], dtype=Kzz.dtype)
    return Kzz


# u are in the kernel space, f in the input space
@Kuf.register(KernelSpaceInducingPoints, DeepKernel, TensorLike)
def Kuf_deep_kernel(inducing_variable, kernel, a_input):
    # print('CALLING DeepKernelGPs Kuf')
    return kernel.basekern(inducing_variable.Z, kernel.cnn(a_input, training=False))


#################################################################################


# 2. InvDeepKernelGP: InvKernel <- DeepKernel <- RBFKernel
# u have been fed through the CNN
@Kuu.register(ConvolvedKernelSpaceInducingPoints, Invariant)
def Kuu_invariant_deep_kernel(inducing_variable, kernel, jitter=0.0):
    # print('CALLING InvDeepKernelGPs Kuu')
    Kzz = kernel.basekern.basekern.K(inducing_variable.Z)
    Kzz += jitter * tf.eye(inducing_variable.Z.shape[0], dtype=Kzz.dtype)
    return Kzz


@Kuf.register(ConvolvedKernelSpaceInducingPoints, Invariant, TensorLike)
def Kuf_invariant_deep_kernel(inducing_variable, kern, Xnew):
    #print('CALLING InvDeepKernelGPs Kuf')
    N, M = tf.shape(Xnew)[0], tf.shape(inducing_variable.Z)[0]
    new_image_shape = Xnew.shape.as_list()
    #print(image_shape)
    new_image_shape[0] = -1
    orbit_size = kern.orbit.orbit_size 
    if kern.orbit.orbit_size is np.inf:
        orbit_size = kern.orbit.minibatch_size
    # print('Xnew', Xnew.shape)
    Xorbit = kern.orbit(Xnew)  # [N, orbit_size, D]
    # print(Xorbit.shape[-1])
    # print('X orbit:', Xorbit.shape)
    #import matplotlib.pyplot as plt
    #plt.imshow(tf.reshape(Xorbit, (-1, 28, 28, 1)).numpy()[0, :, :, 0])
    #plt.show()
    #exit()
    Kzx_orbit = kern.basekern.basekern.K(inducing_variable.Z, kern.basekern.cnn(tf.reshape(Xorbit, new_image_shape), training=False)) # Xorbit.shape[-1]))))  # [M, N * orbit_sz]
    #print('Kzx orbit shape:', Kzx_orbit.shape)
    #print('orbit_size is', orbit_size, 'M is', M, 'N is', N)
    Kzx = tf.reduce_mean(tf.reshape(Kzx_orbit, (M, N, orbit_size)), [2])
    # print('Kzx shape:', Kzx.shape)
    # print(Kzx)
    return Kzx


# 2. InvDeepKernelGP: InvKernel <- DeepKernel <- RBFKernel
# u have been fed through the CNN
@Kuu.register(StochasticConvolvedKernelSpaceInducingPoints, StochasticInvariant)
def Kuu_stochastic_invariant_deep_kernel(inducing_variable, kernel, jitter=0.0):
    # print('CALLING Stoch. InvDeepKernelGPs Kuu')
    # return: [M, N * orbit_sz]
    Kzz = kernel.basekern.basekern.K(inducing_variable.Z)
    Kzz += jitter * tf.eye(inducing_variable.Z.shape[0], dtype=Kzz.dtype)
    return Kzz


# same but for stochastic orbit
@Kuf.register(StochasticConvolvedKernelSpaceInducingPoints, StochasticInvariant, object)
def Kuf_stochastic_invariant_deep_kernel(inducing_variable, kern, Xnew):
    # print('Xnew shape', Xnew.shape)
    # print('CALLING Stoch. InvDeepKernelGPs Kuf')
    # return: [M, N, minibatch_size]
    N, M = tf.shape(Xnew)[0], tf.shape(inducing_variable.Z)[0]
    new_image_shape = Xnew.shape.as_list()
    #print(image_shape)
    new_image_shape[0] = -1
    #print(image_shape)
    minibatch_size = kern.orbit.minibatch_size
    Xorbit = tf.reshape(kern.orbit(Xnew), (N * kern.orbit.minibatch_size, -1))  # [N * minibatch_size, D]
    # print('orbit shape:', Xorbit.shape)
    #import matplotlib.pyplot as plt
    #plt.imshow(tf.reshape(Xorbit, (-1, 28, 28, 1)).numpy()[0, :, :, 0])
    #plt.show()
    #exit()
    Kzx = kern.basekern.basekern.K(inducing_variable.Z, kern.basekern.cnn(tf.reshape(Xorbit, new_image_shape), training=False))    #kern.basekern.cnn(Xorbit))  # [M, N * minibatch_size]
    # print(Kzx.shape, M, N, minibatch_size)
    return tf.reshape(Kzx, (M, N, minibatch_size))
