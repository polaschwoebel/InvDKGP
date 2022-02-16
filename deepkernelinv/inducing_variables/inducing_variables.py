from gpflow.inducing_variables import InducingPoints
from invgp.inducing_variables import ConvolvedInducingPoints, StochasticConvolvedInducingPoints


# 1. DeepKernel inducing points
class KernelSpaceInducingPoints(InducingPoints):
    pass


# 2. a) InvariantDeepKernel inducing points, deterministic orbit (i.e. quantized)
class ConvolvedKernelSpaceInducingPoints(KernelSpaceInducingPoints, ConvolvedInducingPoints):
    pass


# 2. b) InvariantDeepKernel inducing points, stochastic orbit (i.e. learned)
class StochasticConvolvedKernelSpaceInducingPoints(KernelSpaceInducingPoints, StochasticConvolvedInducingPoints):
    pass
