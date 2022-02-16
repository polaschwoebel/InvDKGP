import os
from experiment_scripts.test import test
import tensorflow as tf
from utils import command_line_parser, helpers
from gpflow import config
import numpy as np

parser = command_line_parser.create_parser()
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU

args.autotune = tf.data.experimental.AUTOTUNE  # something about buffering of the batches

args.image_shape = (28, 28, 1)
if args.dataset == 'rotMNIST':
    args.total_nr_data = 12000
elif args.dataset == 'MNIST':
    args.total_nr_data = 60000
elif args.dataset == 'CIFAR10':
    args.total_nr_data = 50000
    args.image_shape = (32, 32, 3)
elif args.dataset == 'PCAM':  
    args.total_nr_data = 262144
    args.nr_classes = 2
    args.image_shape = (96, 96, 3)
elif args.dataset == 'PCAM32':  
    args.total_nr_data = 262144
    args.nr_classes = 2
    args.image_shape = (32, 32, 3)

if args.subset_size is not None:
    args.total_nr_data = args.subset_size

args.image_size = tf.math.reduce_prod(args.image_shape)  #args.image_shape[1:])

model_name = helpers.get_model_name(args)
args = helpers.set_paths(args, model_name)

print('args is', args)

# set precision
if args.precision == 'single':
    config.set_default_float(np.float32)
    tf.keras.backend.set_floatx('float32')
    config.set_default_int(np.int32)
elif args.precision == 'double':
    config.set_default_float(np.float64)
    config.set_default_int(np.int64)
    tf.keras.backend.set_floatx('float64')
else:
    print('pass valid precision')
    exit()

# training
if args.mode == 'train':
    if args.use_model == 'CNN':
        from experiment_scripts.CNN_training import train
        train(args)
    elif 'GP' in args.use_model:
        from experiment_scripts.SVGP_training import train
        train(args)
    else:
        print('Pass valid model.')
        exit()

if args.mode == 'test':
    test(args)

