import gpflow
import numpy as np
import numpy.random as rnd
import pytest
import tensorflow as tf
from gpflow.models import SVGP
from gpflow.utilities import read_values, multiple_assign, set_trainable

import invgp
from deepkernelinv.kernels import DeepKernel
from invgp.inducing_variables.invariant_convolution_domain import StochasticConvolvedInducingPoints
from invgp.models.SampleSVGP import SampleSVGP

import matplotlib.pyplot as plt
from gpflow.config import default_float

np.random.seed(0)


# Test: use deep kernel with identity function to see that it's the same as the non deep kernel
def test_deep_kernel():
    # generate datapoints
    X = np.random.uniform(-3, 3, 400)[:, None]
    X = np.reshape(X, [200, 2])  # 2-dimensional input
    Y = np.ones((200, 1))
    Y[np.where(np.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2) > 2)] = 0
    train_dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    train_dataset = train_dataset.shuffle(1024).batch(len(X))

    # plot everything
    # fig, ax = plt.subplots(1, 1, figsize=(12, 15))
    # x1list = np.linspace(-3.0, 3.0, 100)
    # x2list = np.linspace(-3.0, 3.0, 100)
    # X1, X2 = np.meshgrid(x1list, x2list)
    # ax.set_title('True function')
    # ax.set_aspect('equal', 'box')
    # # plot the true data generating process
    # true_Y = np.ones_like(X1)
    # true_Y[np.where(np.sqrt(X1 ** 2 + X2 ** 2) > 2)] = 0
    # cp = ax.contourf(X1, X2, true_Y)
    # plt.colorbar(cp)
    #plt.show()

    # initialize SVGP model
    inducing_variables = X.copy()
    SVGP_model = SVGP(
            gpflow.kernels.SquaredExponential(),
            gpflow.likelihoods.Gaussian(),
            inducing_variable=inducing_variables,
            num_data=len(X))

    # initialize DeepKernel models
    cnn = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(2,)),
        tf.keras.layers.Lambda(lambda x: tf.cast(x, default_float()))])  # input_shape)])      

    deepkernel = DeepKernel((2, 1),
                    filters_l1=0, 
                    filters_l2=0, 
                    hidden_units=0, 
                    output_units=2,
                    basekern=gpflow.kernels.SquaredExponential(),
                    batch_size=200, cnn=cnn)
    deepkernel_SVGP_model = SVGP(
            deepkernel,
            gpflow.likelihoods.Gaussian(),
            inducing_variable=inducing_variables,
            num_data=len(X))

    # check that untrained ELBOs are the same
    untrained_elbo = SVGP_model.elbo((X, Y)).numpy()
    untrained_deepkernel_elbo = deepkernel_SVGP_model.elbo((X, Y)).numpy()
    np.testing.assert_allclose(untrained_elbo, untrained_deepkernel_elbo, rtol=0.01, atol=0.0)

    train_iter = iter(train_dataset.repeat())
    training_loss = SVGP_model.training_loss_closure(train_iter, compile=True)
    optimizer = tf.keras.optimizers.Adam(0.01)
    set_trainable(SVGP_model.inducing_variable, False)

    @tf.function
    def optimization_step():
        optimizer.minimize(training_loss, SVGP_model.trainable_variables)

    elbo_hist = []
    for step in range(2000):
        optimization_step()
        if step % 10 == 0:
            minibatch_elbo = -training_loss().numpy()
            print("Step: %s, Mini batch elbo: %s" % (step, minibatch_elbo))
            elbo_hist.append(minibatch_elbo)


    # initialize deepkernel SVGP model with fitted parameters from SVGP
    trained_params = read_values(SVGP_model)

    def key_mapper(key):
        if 'kernel' in key:
            key = key.replace('kernel', 'kernel.basekern')
        return key

    trained_params = {key_mapper(key): value for key, value in trained_params.items()}

    multiple_assign(deepkernel_SVGP_model, trained_params)
    gpflow.utilities.print_summary(SVGP_model)
    gpflow.utilities.print_summary(deepkernel_SVGP_model)

    # check that trained elbos are close
    trained_elbo = SVGP_model.elbo((X, Y)).numpy()
    trained_deepkernel_elbo = deepkernel_SVGP_model.elbo((X, Y)).numpy()
    np.testing.assert_allclose(trained_elbo, trained_deepkernel_elbo, rtol=0.01, atol=0.0)


    # TODO, other test: train 'real' deep kernel to see that it can fit the data better