import tensorflow as tf
from datetime import datetime


def extract_numpy(value):
    if tf.is_tensor(value):
        return value.numpy()
    else:
        return value


def setup_tensorboard():
    train_data_dir = "visualizations/train_data/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    train_file_writer = tf.summary.create_file_writer(train_data_dir)
    test_data_dir = "visualizations/test_data/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    test_file_writer = tf.summary.create_file_writer(test_data_dir)
    return train_data_dir, test_data_dir, train_file_writer, test_file_writer


def inspect_conditional(inducing_variable_type, kernel_type):
    """
    Helper function returning the exact implementation called
    by the multiple dispatch `conditional` given the type of
    kernel and inducing variable.

    :param inducing_variable_type:
        Type of the inducing variable
    :param kernel_type:
        Type of the kernel

    :return: String
        Contains the name, the file and the linenumber of the
        implementation.
    """
    import inspect
    from gpflow.conditionals import conditional

    implementation = conditional.dispatch(object, inducing_variable_type, kernel_type, object)
    info = dict(inspect.getmembers(implementation))
    return info["__code__"]


def get_model_name(args):
    # base name
    model_name = '%s_%s' % (
        args.use_model, args.dataset
        )
    if args.subset_size is not None:
        model_name = '%s_%s_%s' % (
            args.use_model, args.dataset, args.subset_size)

    if args.deepkernel:
        model_name += '_deepk'

    if 'GP' in args.use_model:
        model_name += '_%s' %args.ind_point_init
        if args.experiment_config != 'vanilla':
            model_name += '_' + args.experiment_config
        if args.fix_kernel_variance:
            model_name += '_fix_kernel_var'
        if args.ARD:
            model_name += '_ARD'
        if args.freeze_CNN:
            model_name += '_freeze_CNN'
        if args.use_model != 'GPR':
            model_name += '_ind_points_%s' % args.nr_inducing_points
        model_name += '_%s' % args.likelihood

        if args.q_sqrt_init != 0.01:
            model_name += 'q_sqrt_%s' %args.q_sqrt_init
        # model_name += '_likelihd_var_%s' % args.likelihood_variance
        if args.fix_lengthscale:
            model_name += '_fix_lengthsc'
    else:
        if args.use_model == 'CNN':
            model_name += '_%s' % args.CNN_architecture
        if args.dropout_prob != 0.2:
            model_name += '_dropout_%s' % args.dropout_prob

    if args.pretrained_CNN_path is not None:
        model_name += '_pretrain_cnn'

    # invariance params
    if args.invariant:
        model_name += '_%s' % args.use_orbit
        # model_name += '_use_stn=%s' % args.use_stn
        model_name += '_orbits_%s' % ('_').join([str(val) for val in args.orbit_size])
        if args.fix_angle:
            model_name += '_fix_angle'
            model_name += '_' + args.angle
        # if args.angle is not None:
        #     model_name += '_angle_%s' %args.angle
        if args.use_orbit == 'seven_param_affine':
            if args.radians:
                model_name += '_rad'
        if args.affine_init > 0:
            model_name += '_init_%s' % args.affine_init
        if args.fix_orbit:
            model_name += '_fix_orb'
            model_name += '_' + str(args.angle) #hard coded for PCAM

    if args.fix_likelihood:
        model_name += '_fix_lkh'
    if args.fix_orbit_initial:
        model_name += 'fix_orbit_initial'
    if args.matheron:
        model_name += '_matheron'
    if not args.num_samples == 5:
        model_name += '_num_samples_%s' % args.num_samples
    if args.fix_indp:
        model_name += '_fix_indp'

    # if args.learning_rate != 0.001:
    model_name += '_lr_%s' % ('_').join([str(val) for val in args.learning_rate])

    if args.decay_lr != ['none', 'none']:
        model_name += '_decay_lr_%s' % ('_').join([str(val) for val in args.decay_lr])

    if args.use_model != 'GPR':
        model_name += '_batchs_%s' % ('_').join([str(val) for val in args.batch_size])

    if args.run != 0:
        model_name += '_run_%s' %args.run

    if args.optimizer != 'adam':
        model_name += '_%s' % ('_').join([str(val) for val in args.optimizer])
    if args.DA_color:
        model_name += '_DA_color_%s_%s' % (args.DA_params_contrast[1], args.DA_params_brightness[1])
    if args.DA_affine:
        model_name += '_DA_affine_%s' % args.DA_params_affine 
    if args.DA_rotate:
        model_name += '_DA_rotate_%s' % args.radian_init
    if args.DA_rot_quant:
        model_name += '_DA_rot_quant'
    if args.DA_mirror_translate:
        model_name += '_DA_mirror_translate'
    if args.DA_mirror_rotate:
        model_name += '_DA_mirror_rotate'
    if args.DA_translate:
        model_name += '_DA_translate'
    if args.DA_mirror_translate_rotate:
        model_name += '_DA_mirror_translate_rotate'
    if args.DA_mirror_translate_rotate or args.DA_mirror_translate or args.DA_translate:
        model_name += '_%s' % args.DA_pad

    #if args.latent_dim != 50:
    model_name += '_ltdm_%s' % args.latent_dim

    if args.CNN_architecture == 'MLE_learned_da':
        model_name += '_MLE_learned_da'
        model_name += '_init_%s' %args.affine_init

    if args.name_appendix is not None:
        model_name += args.name_appendix

    return model_name


def set_paths(args, model_name):
    path = args.path
    # numpy logging
    train_results_path = path + model_name + '_train_results.p'
    test_results_path = path + model_name + '_test_results.p'
    model_params_path = path + model_name + '_model_params.p'
    args.train_results_path = train_results_path
    args.test_results_path = test_results_path
    args.model_params_path = model_params_path
    # tensorboard loggig
    tensorboard_logdir = path + model_name
    args.tensorboard_logdir = tensorboard_logdir
    # model saving
    model_path = path + 'model_%s' % model_name
    args.model_path = model_path
    return args


# helpers for pretrained CNN + MLE DA
def increment_weightname(string):
    start, mid, end = string.split('[')
    number, suffix = mid.split(']')
    decremented_number = int(number) + 1
    new_string = start + '[' + str(decremented_number) + ']' + suffix + '[' + str(end)
    return new_string


def increment_biasname(string):
    start, end = string.split('[')
    number, suffix = end.split(']')
    decremented_number = int(number) + 1
    new_string = start + '[' + str(decremented_number) + ']' + suffix
    return new_string


def name_mapper(string):
    if string.endswith(']'):
        new_string = increment_weightname(string)
    else:
        new_string = increment_biasname(string)
    return new_string
