from .gpflow_logger import CustomModelToTensorBoard
from .set_trainable_params import *
from .coord_ascent_helpers import *


def toggle_learnable_params(args, model):
    if args.deepkernel and args.invariant:
        toggle_learnable_params_deep_inv(args, model)
    elif args.deepkernel:
        toggle_learnable_params_noninv_deep(args, model)
    else:
        print('Coord. ascent not implemented for shallow kernels.')
        exit()


def initialize_with_trained_params(args, X_train, Y_train, X_test, Y_test, initialize='hyperparams'):
    if args.use_model == 'GP':
    	from ..load_pretraiend import initialize_with_SGPR_params as initialize
    if args.use_model == 'InvGP':
        from ..load_pretrained import initialize_with_quantized_params as initialize
    print('INITIALIZING USING ARGS:', args)
    model = initialize(args, X_train, Y_train, X_test, Y_test, initialize='hyperparams')
    return model
