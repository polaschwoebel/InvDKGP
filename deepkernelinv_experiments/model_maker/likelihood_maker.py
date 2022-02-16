import gpflow
from gpflow.utilities import set_trainable


def initialize_likelihood_function(args):
    if args.likelihood == 'Gaussian':
        likelihood = gpflow.likelihoods.Gaussian()
        likelihood.variance.assign(args.likelihood_variance)
        if args.fix_likelihood:
            set_trainable(likelihood.variance, False)
    elif args.likelihood == 'Softmax':
        likelihood = gpflow.likelihoods.Softmax(args.nr_classes)
    elif args.likelihood == 'Robustmax':
        invlink = gpflow.likelihoods.RobustMax(args.nr_classes)  # Robustmax inverse link function
        likelihood = gpflow.likelihoods.MultiClass(args.nr_classes, invlink=invlink) 
    return likelihood
