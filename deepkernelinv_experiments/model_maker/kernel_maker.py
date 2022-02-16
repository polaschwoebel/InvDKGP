import gpflow
from gpflow.utilities import set_trainable


def initialize_basekern(args):
    if args.ARD:
        kernel = gpflow.kernels.SquaredExponential(lengthscales=[args.basekern_lengthscale]*args.latent_dim)
    else:
        kernel = gpflow.kernels.SquaredExponential()
        kernel.lengthscales.assign(args.basekern_lengthscale)
    kernel.variance.assign(args.basekern_variance)
    if args.fix_kernel_variance:
        set_trainable(kernel.variance, False)
    if args.fix_lengthscale:
        set_trainable(kernel.lengthscales, False)
    return kernel
