from gpflow.utilities import set_trainable
from tensorflow.keras import backend

# unified coord ascent helper - deep, invariant kernel 
def toggle_learnable_params_deep_inv(args, model):
    # from CNN to GP layer
    if not model.q_mu.trainable:  # use q_mu as "GP-phase indicator"
        print('\n Toggle training from CNN to GP layer.')
        set_trainable(model.kernel.basekern.cnn, False)
        set_trainable(model.kernel.basekern.basekern.lengthscales, True)
        set_trainable(model.inducing_variable.Z, True)
        set_trainable(model.q_mu, True)
        set_trainable(model.q_sqrt, True)

        # make sure fixed_parameters stay fixed
        if not args.fix_orbit:
            set_trainable(model.kernel.orbit, True)
        if args.likelihood == 'Gaussian' and not args.fix_likelihood:
            set_trainable(model.likelihood.variance, True)
        if not args.fix_kernel_variance:
            set_trainable(model.kernel.basekern.basekern.variance, True)
        return

    # from GP layer to CNN
    if model.q_mu.trainable:
        print('\n Toggle training from GP layer to CNN.')
        set_trainable(model.kernel.basekern.cnn, True)
        set_trainable(model.kernel.basekern.basekern.variance, False)
        set_trainable(model.kernel.basekern.basekern.lengthscales, False)
        set_trainable(model.kernel.orbit, False)
        set_trainable(model.inducing_variable.Z, False)
        set_trainable(model.q_mu, False)
        set_trainable(model.q_sqrt, False)
        if args.likelihood == 'Gaussian':
            set_trainable(model.likelihood.variance, False)


# unified coord ascent helper - non-inv. + deep kernel 
def toggle_learnable_params_noninv_deep(args, model):
    # from CNN to GP layer
    if not model.q_mu.trainable:  # use q_mu as "GP-phase indicator"
        print('\n Toggle training from CNN to GP layer.')
        set_trainable(model.kernel.cnn, False)
        set_trainable(model.kernel.basekern.lengthscales, True)
        set_trainable(model.inducing_variable.Z, True)
        set_trainable(model.q_mu, True)
        set_trainable(model.q_sqrt, True)

        # make sure fixed_parameters stay fixed
        if args.likelihood == 'Gaussian' and not args.fix_likelihood:
            set_trainable(model.likelihood.variance, True)
        if not args.fix_kernel_variance:
            set_trainable(model.kernel.basekern.variance, True)
        return

    # from GP layer to CNN
    if model.q_mu.trainable:
        print('\n Toggle training from GP layer to CNN.')
        set_trainable(model.kernel.cnn, True)
        set_trainable(model.kernel.basekern.variance, False)
        set_trainable(model.kernel.basekern.lengthscales, False)
        set_trainable(model.inducing_variable.Z, False)
        set_trainable(model.q_mu, False)
        set_trainable(model.q_sqrt, False)
        if args.likelihood == 'Gaussian':
            set_trainable(model.likelihood.variance, False)


###############################################################
def toggle_orbit_batch_size(args, batch_size, map_fn, training_ds, test_ds, model):
    # toggle batch size and create new datasets
    if batch_size == args.batch_size[0]:  # GP training
        batch_size = args.batch_size[1]  # convnet training
    elif batch_size == args.batch_size[1]:
        batch_size = args.batch_size[0]

    train_dataset = training_ds \
        .shuffle(1024) \
        .batch(batch_size, drop_remainder=False) \
        .map(map_fn, num_parallel_calls=args.autotune) \
        .prefetch(args.autotune)
    test_dataset = test_ds\
        .batch(batch_size, drop_remainder=False)\
        .map(map_fn, num_parallel_calls=args.autotune) \
        .prefetch(args.autotune)
    if args.invariant:
        if model.kernel.orbit.minibatch_size == args.orbit_size[0]:
            model.kernel.orbit.minibatch_size = args.orbit_size[1]
        elif model.kernel.orbit.minibatch_size == args.orbit_size[1]:
            model.kernel.orbit.minibatch_size = args.orbit_size[0]

    return train_dataset, test_dataset, batch_size
