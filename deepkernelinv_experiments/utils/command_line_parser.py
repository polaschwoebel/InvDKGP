import argparse


def create_parser():
    parser = argparse.ArgumentParser(description='')
    # data
    parser.add_argument('-dataset', type=str, default='rotMNIST', help='which dataset to use.')
    parser.add_argument('-subset_size', type=int, default=None)
    parser.add_argument('-image_shape', default=(28, 28, 1))
    parser.add_argument('-image_size', default=(28 * 28), type=int)
    parser.add_argument('-nr_classes', type=int, default=10)
    parser.add_argument('-latent_dim', type=int, default=50)

    # model
    parser.add_argument('-use_model', required=True, help='Which model to use?')
    parser.add_argument('-invariant', action='store_true', help='Are we training an invariant model?')
    parser.add_argument('-deepkernel', action='store_true', help='Are we training a deep model?')

    parser.add_argument('-use_orbit', type=str, help='Which orbit to use?') 
    parser.add_argument('-orbit_size', type=int, nargs="+", default=[90, 90])
    parser.add_argument('-use_stn', action='store_true')
    parser.add_argument('-mode', default='train', help='Train or test mode?')
    parser.add_argument('-pretrained_CNN_path', type=str, help='pass model path if pre-trained model should be used.')
    parser.add_argument('-matheron', action='store_true')
    parser.add_argument('-num_samples', type=int, default=5)
    parser.add_argument('-nr_inducing_points', type=int, default=100)
    parser.add_argument('-ind_point_init', type=str, default='inducing-init')  # 'zeros', 'kmeans', 'random_x'
    parser.add_argument('-likelihood', type=str, default='Gaussian')
    parser.add_argument('-precision', type=str, default='single')
    parser.add_argument('-dropout', action='store_true')
    parser.add_argument('-dropout_prob', type=float, default=0.2)  # only used for litjens_vgg at the moment
    parser.add_argument('-CNN_architecture', type=str, default='small')
    parser.add_argument('-radians', type=bool, default=True, help='use radians or degrees in orbit parametrisation? (5 param only)')
    parser.add_argument('-ARD', action='store_true') # use automatic relevance detection in kernel

    # initial parameters
    parser.add_argument('-angle', type=float, default=None)
    parser.add_argument('-affine_init', type=float, default=0.)
    parser.add_argument('-radian_init', type=float, default=0.)
    parser.add_argument('-basekern_lengthscale', type=float, default=10)
    parser.add_argument('-basekern_variance', type=float, default=1)
    parser.add_argument('-likelihood_variance', type=float, default=0.05)
    parser.add_argument('-q_sqrt_init', type=float, default=0.01)

    # training
    parser.add_argument('-nr_epochs', default=100, type=int, help='How many epochs to train for.')
    parser.add_argument('-GPU', default=str(0), type=str, help='Which GPU to use.')
    parser.add_argument('-evaluate_on', type=str, default='test', help='evaluate on test or validaton data?')
    parser.add_argument('-test_frequency', type=int, default=10)
    parser.add_argument('-full_batch_frequency', type=int, default=200)
    parser.add_argument('-minibatch_frequency', type=int, default=10)
    parser.add_argument('-test_during_training', action='store_true')
    parser.add_argument('-experiment_config', type=str, default='vanilla')  # 'pretrain_hyperparams', 'coord_ascent', 'fix_variances', 'fix_variances_partly'
    parser.add_argument('-toggle_after_steps', type=int, default=3000)
    parser.add_argument('-continue_training', action='store_true')
    parser.add_argument('-learning_rate', type=float, nargs="+", default=[0.001, 0.001])
    parser.add_argument('-optimizer', type=str, nargs="+", default=['adam', 'adam'])
    parser.add_argument('-batch_size', type=int, nargs="+", default=[200, 200])   #200)
    parser.add_argument('-path', type=str, default='../experiments/coord_asc_experiments_18_08/')
    parser.add_argument('-run', type=int, default=0)
    parser.add_argument('-decay_lr', type=str, nargs="+", default=['none', 'none'])  # smooth, steps
    parser.add_argument('-DA_color', action='store_true')
    parser.add_argument('-DA_affine', action='store_true')
    parser.add_argument('-DA_rotate', action='store_true')
    parser.add_argument('-DA_rot_quant', action='store_true')
    parser.add_argument('-DA_mirror_translate', action='store_true')
    parser.add_argument('-DA_mirror_rotate', action='store_true')
    parser.add_argument('-DA_translate', action='store_true')
    parser.add_argument('-DA_mirror_translate_rotate', action='store_true')
    parser.add_argument('-DA_pad', type=str, default='zeropad')
    parser.add_argument('-DA_params_contrast', type=float, nargs="+", default=[-0.5, 0.5])
    parser.add_argument('-DA_params_brightness', type=float, nargs="+", default=[-0.5, 0.5])
    parser.add_argument('-DA_params_affine', type=float, default=0.1)
    parser.add_argument('-nr_eval_batches', type=int, default=None)

    # fix certain parameters
    parser.add_argument('-fix_kernel_variance', action='store_true')
    parser.add_argument('-freeze_CNN', action='store_true')
    parser.add_argument('-fix_likelihood', action='store_true')
    parser.add_argument('-fix_indp', action='store_true')
    parser.add_argument('-fix_angle', action='store_true')
    parser.add_argument('-fix_orbit', action='store_true')
    parser.add_argument('-fix_orbit_initial', action='store_true')
    parser.add_argument('-fix_lengthscale', action='store_true')  

    # name appendix to bookkeep other stuff
    parser.add_argument('-name_appendix', type=str, default=None)  

    return parser
