# tf.config.experimental_run_functions_eagerly(True)


def create_model(args, X=None, Y=None):

    # initalize model based on model type
    if args.use_model == 'SGPR':
        from .SGPR import create_model
        model = create_model(args, X, Y)
    elif args.use_model == 'GPR':
        from .GPR import create_model
        model = create_model(args, X, Y)

    else:
        if args.use_model == 'CNN':
            from .CNN import create_model
        elif args.use_model == 'SampleSVGP':
            if args.deepkernel:
                from .DeepKernelGP import create_model
            else:
                from .SampleSVGP import create_model
        elif 'SVGP' in args.use_model:
            if args.deepkernel:
                from .DeepKernelGP import create_model
            else:
                from .GP import create_model
        else:
            raise ValueError('Unsupported model: {}!'.format(args.use_model))

    model = create_model(args, X)
    return model


def create_optimizer(args):
    if args.experiment_config == 'coord_ascent_training': 
        # in this case there are two learning rates + decay options, one for each training phase 
        iterations = args.toggle_after_steps
        # start by training GP
        decay_lr = args.decay_lr[0]
        learning_rate = args.learning_rate[0]
        use_optimizer = args.optimizer[0]

    else:
        iterations = args.nr_epochs * (args.total_nr_data // args.batch_size[0])
        decay_lr = args.decay_lr[0]
        learning_rate = args.learning_rate[0]
        use_optimizer = args.optimizer[0]

    from .optimizer_maker import create_optimizer
    optimizer = create_optimizer(args, use_optimizer, learning_rate, decay_lr, iterations)

    return optimizer


def create_map_fn(args):
    # initalize map_fn based on model type
    if args.dataset in ['rotMNIST_extracted_features', 'rotMNIST_extracted_features_normalized', 'QM9']:
        from .map_functions import identity as map_fn
        return map_fn

    else:
        if args.use_model == 'CNN':  # square
            from .map_functions import map_fn_image_onehot as map_fn

        elif args.deepkernel:
            if args.likelihood == 'Gaussian': # square, onehot
                from .map_functions import map_fn_image_onehot as map_fn
            if args.likelihood == 'Softmax': # square, int
                from .map_functions import map_fn_image_int as map_fn

        elif args.use_model != 'CNN' and not args.deepkernel:
            if args.likelihood == 'Gaussian': # flat, onehot
                from .map_functions import map_fn_flat_onehot as map_fn
            if args.likelihood == 'Softmax': # flat, int
                from .map_functions import map_fn_flat_int as map_fn

        else:
            raise ValueError('Unsupported model: {}!'.format(args.use_model))

        return lambda image, label: map_fn(image, label, args.nr_classes)


def initialize_inducing_points(args, train_dataset, method='kmeans', kernel=None, orbit=None):
    if 'GP' in args.use_model:
        from .ind_point_maker import initialize_inducing_points
        inducing_points = initialize_inducing_points(args, train_dataset, method=method, kernel=kernel, orbit=orbit)
    elif args.use_model == 'CNN':
        inducing_points = None

    else:
        raise ValueError('Unsupported model: {}!'.format(args.use_model))
    return inducing_points


def make_evaluation_step(args, batch, model):
    if 'GP' in args.use_model:
        if args.likelihood == 'Gaussian':
            from .GP import gaussian_evaluation_step as evaluation_step
        elif args.likelihood == 'Softmax' or args.likelihood == 'Robustmax':
            from .GP import softmax_evaluation_step as evaluation_step
    elif args.use_model == 'CNN' or args.use_model == 'AE':
        from .CNN import evaluation_step
    batch_acc = evaluation_step(args.nr_classes, batch, model)
    return batch_acc
