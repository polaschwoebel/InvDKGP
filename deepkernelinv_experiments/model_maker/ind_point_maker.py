import tensorflow as tf
from scipy.cluster.vq import kmeans2
import numpy as np
import numpy.random as rnd
import robustgp
from .kernel_maker import initialize_basekern
from .orbit_maker import initialize_orbit
from gpflow.config import default_float
from utils import load_datasets


def initialize_inducing_points(args, train_dataset, method='kmeans', kernel=None, orbit=None):
    if args.mode == 'test':
        inducing_variables = np.zeros((args.nr_inducing_points, args.latent_dim), dtype=default_float())
        return inducing_variables

    print('Initializing inducing points using', method)
    #########
    # create orbits and CNN features in deep/invariant case
    invariant_init = args.invariant and (args.affine_init != 0. or args.radian_init != 0. or args.angle is not None)
    print('invariant init is', invariant_init)

    if not args.deepkernel and not invariant_init:
        X_train = load_datasets.create_full_batch(train_dataset)
        initkernel = kernel

    if not args.deepkernel and invariant_init:
        print('creating orbits for invariant ind. points')
        orbit = initialize_orbit(args, samples=3)
        nr_datapoints = 5000  # 200 if args.dataset in 'CIFAR10' else 5000
        X_train = load_datasets.create_full_batch(train_dataset, nr_datapoints)
        print('X train shape', X_train.shape)
        X_train_subset = X_train[rnd.permutation(len(X_train))[:nr_datapoints], :]
        print('X_train subset shape', X_train_subset.shape)
        X_orbits = orbit(X_train_subset)
        print('X orbit shape', X_orbits.shape)
        X_train = tf.reshape(X_orbits, [-1, args.image_shape[0], args.image_shape[1], args.image_shape[2]]).numpy()
        print('reshaped orbit shape', X_train.shape)
        initkernel = kernel.basekern

    if args.deepkernel and not invariant_init:
        print('feeding data through CNN for indp. init.')
        all_features = []
        for i, batch in enumerate(train_dataset):
            features = kernel.cnn(batch[0], training=False)
            all_features.append(features)
        X_train = tf.experimental.numpy.vstack(all_features)
        initkernel = kernel.basekern

    if args.deepkernel and invariant_init:
        orbit = initialize_orbit(args, samples=3)
        print('feeding data through CNN for indp. init.')
        all_features = []
        for i, batch in enumerate(train_dataset):
            if i % 100 == 0:
                print('processing batch', i)
            X_orbits = orbit(batch[0])
            X_orbits = tf.reshape(X_orbits, [-1, args.image_shape[0], args.image_shape[1], args.image_shape[2]])
            features = kernel.cnn(X_orbits, training=False)
            all_features.append(features)
        X_train = tf.experimental.numpy.vstack(all_features)
        initkernel = kernel.basekern

    #########
    # initialise indp. with different methods
    if method == 'zeros':
        inducing_variables = np.zeros((args.nr_inducing_points, args.image_size), dtype=default_float())

    elif method == 'kmeans':
        # X_train = np.reshape(X_train, (-1, args.image_size))
        inducing_variables = kmeans2(X_train.numpy(),
                                     args.nr_inducing_points,
                                     minit='points')[0]
        # X_train = np.reshape(X_train, (-1, args.image_size))

    elif method == 'random_x':
        inducing_variables = X_train.numpy()[rnd.permutation(len(X_train))[:args.nr_inducing_points], :]

    elif method == 'inducing-init' or method == 'reinit':
        initializer = robustgp.ConditionalVariance()

        inducing_variables, _ = initializer.compute_initialisation(
            X_train.numpy(), args.nr_inducing_points, initkernel)

    else:
        print('Pass valid method.')
        exit()

    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(2)
    # ax = ax.flatten()
    # for i in range(20):
    #    ax[0].plot(inducing_variables[i])
    #    ax[1].plot(X_train[i])
    # plt.show()
    # exit()

    return inducing_variables
