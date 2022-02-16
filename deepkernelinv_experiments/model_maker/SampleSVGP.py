import tensorflow as tf
import gpflow
from invgp.kernels import Invariant, StochasticInvariant
from invgp.models import SampleSVGP
from invgp.inducing_variables.invariant_convolution_domain import ConvolvedInducingPoints, StochasticConvolvedInducingPoints
from .kernel_maker import initialize_basekern
from .likelihood_maker import initialize_likelihood_function
from .orbit_maker import initialize_orbit
from gpflow.config import default_float
from .ind_point_maker import initialize_inducing_points


def create_model(args, X_train):
    likelihood = initialize_likelihood_function(args)
    kernel = initialize_basekern(args)
    inducing_variable = initialize_inducing_points(
             args, X_train, method=args.ind_point_init, kernel=kernel)

    if args.invariant:
        orbit = initialize_orbit(args)
        if args.use_orbit in ['sampled_rot', 'general_affine', 'seven_param_affine', 'colorspace', 'inverse_angle_rot']:
            print('using stochastic framework')
            inducing_variable = StochasticConvolvedInducingPoints(inducing_variable)
            kernel = StochasticInvariant(
                basekern=kernel,
                orbit=orbit)
        else:
            print('using deterministic framework')
            inducing_variable = ConvolvedInducingPoints(inducing_variable)
            kernel = Invariant(
                basekern=kernel,
                orbit=orbit)

    model = SampleSVGP(kernel, likelihood,
                               inducing_variable=inducing_variable,
                               num_data=args.total_nr_data,
                               num_latent_gps=args.nr_classes,
                               matheron_sampler = args.matheron,
                               num_samples = args.num_samples)

    model.q_sqrt.assign(model.q_sqrt / 100)

    return model
