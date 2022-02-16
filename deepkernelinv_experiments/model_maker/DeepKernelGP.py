import gpflow
import invgp
from deepkernelinv.inducing_variables import KernelSpaceInducingPoints, ConvolvedKernelSpaceInducingPoints, StochasticConvolvedKernelSpaceInducingPoints
from deepkernelinv_experiments.utils.helpers import inspect_conditional
from .kernel_maker import initialize_basekern
from .likelihood_maker import initialize_likelihood_function
from .orbit_maker import initialize_orbit
from .CNN import initialize_cnn, initialize_deep_kernel_cnn, load_pretrained_cnn, load_pretrained_encoder
from .ind_point_maker import initialize_inducing_points

from gpflow.utilities import set_trainable, print_summary
from deepkernelinv.kernels import DeepKernel
from deepkernelinv.mean_functions import ZeroImage
# tf.config.experimental_run_functions_eagerly(True)


def create_model(args, X_train):
    # Gaussian likelihood
    if args.pretrained_CNN_path is not None:
        cnn = load_pretrained_cnn(args) 
    else:
        cnn = initialize_deep_kernel_cnn(args)
    likelihood = initialize_likelihood_function(args)
    basekern = initialize_basekern(args)
    deep_kernel = DeepKernel(
        cnn,
        args.image_shape,
        basekern=basekern)

    inducing_points = initialize_inducing_points(
             args, X_train, method=args.ind_point_init, kernel=deep_kernel)

    if args.invariant:
        orbit = initialize_orbit(args)
        if args.use_orbit == 'quantized_rot':  # deterministic orbit
            inducing_points = ConvolvedKernelSpaceInducingPoints(inducing_points)
            kernel = invgp.kernels.Invariant(
                basekern=deep_kernel,
                orbit=orbit)
            print('inspect conditional I yields', inspect_conditional(
                ConvolvedKernelSpaceInducingPoints, invgp.kernels.Invariant))

        elif args.use_orbit in ['sampled_rot', 'general_affine', 'seven_param_affine', 'colorspace', 'inverse_angle_rot']:  # stochastic orbit
            inducing_points = StochasticConvolvedKernelSpaceInducingPoints(inducing_points)
            kernel = invgp.kernels.StochasticInvariant(
                basekern=deep_kernel,
                orbit=orbit)
            print('inspect conditional II yields', inspect_conditional(
                StochasticConvolvedKernelSpaceInducingPoints, invgp.kernels.StochasticInvariant))

        else:
            print('Please pass valid orbit.')
            exit()

    else:
        kernel = deep_kernel
        # inducing points in feature space/kernel input space, i.e. mapped through CNN
        inducing_points = KernelSpaceInducingPoints(inducing_points)

    if args.use_model == 'SVGP':
        model = gpflow.models.SVGP(kernel, likelihood,
                                   inducing_variable=inducing_points,
                                   num_data=args.total_nr_data,
                                   num_latent_gps=args.nr_classes,
                                   mean_function=ZeroImage(args.nr_classes))  # always zero mean
    if args.use_model == 'SampleSVGP':
        model = invgp.models.SampleSVGP(kernel, likelihood,
                                        inducing_variable=inducing_points,
                                        num_data=args.total_nr_data,
                                        num_latent_gps=args.nr_classes,
                                        matheron_sampler=args.matheron,
                                        num_samples=args.num_samples,
                                        mean_function=ZeroImage(args.nr_classes))

    model.q_sqrt.assign(model.q_sqrt * args.q_sqrt_init)

    # fix parameters if desired
    if args.freeze_CNN:
        if args.invariant and args.deepkernel:
            set_trainable(model.kernel.basekern.cnn, False)
        elif args.deepkernel:
            set_trainable(model.kernel.cnn, False)
    if args.fix_likelihood:
        set_trainable(model.likelihood, False)

    return model
