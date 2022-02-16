import tensorflow as tf
import gpflow
import invgp
from invgp.inducing_variables.invariant_convolution_domain import ConvolvedInducingPoints, StochasticConvolvedInducingPoints
from .kernel_maker import initialize_basekern
from .likelihood_maker import initialize_likelihood_function
from .orbit_maker import initialize_orbit
from gpflow.config import default_float, default_int
from deepkernelinv.mean_functions import ZeroImage
from .ind_point_maker import initialize_inducing_points


@tf.function(autograph=False)
def softmax_evaluation_step(nr_classes, batch, model):
    images = batch[0]
    labels = batch[1]
    m, v = model.predict_y(images)
    preds = tf.reshape(tf.argmax(m, 1, output_type=default_int()), labels.shape)
    correct = (preds == tf.cast(labels, default_int()))
    acc = tf.reduce_mean(tf.cast(correct, default_float()))
    return acc


@tf.function(autograph=False)
def gaussian_evaluation_step(nr_classes, batch, model):
    images = batch[0]
    labels = batch[1]
    m, v = model.predict_f(images)
    m_argmax = tf.argmax(m, 1)
    preds = tf.one_hot(tf.cast(m_argmax, tf.uint8), nr_classes)
    preds = tf.cast(preds, default_int())
    correct = tf.reduce_all(tf.equal(preds, tf.cast((labels), default_int())), axis=1)
    acc = tf.reduce_mean(tf.cast(correct, default_float()))
    return acc


def create_model(args, X_train):
    likelihood = initialize_likelihood_function(args)
    kernel = initialize_basekern(args)
    inducing_variable = initialize_inducing_points(
             args, X_train, method=args.ind_point_init, kernel=kernel)


    if args.invariant:
        orbit = initialize_orbit(args)
        if args.use_orbit == 'quantized_rot':
            print('using deterministic framework')
            inducing_variable = ConvolvedInducingPoints(inducing_variable)
            kernel = invgp.kernels.Invariant(
                basekern=kernel,
                orbit=orbit)
        else:
            print('using stochastic framework')
            inducing_variable = StochasticConvolvedInducingPoints(inducing_variable)
            kernel = invgp.kernels.StochasticInvariant(
                basekern=kernel,
                orbit=orbit)


    model = gpflow.models.SVGP(kernel, likelihood,
                               inducing_variable=inducing_variable,
                               num_data=args.total_nr_data,
                               num_latent_gps=args.nr_classes, 
                               mean_function=ZeroImage(args.nr_classes))
    model.q_sqrt.assign(model.q_sqrt / 100)

    return model
