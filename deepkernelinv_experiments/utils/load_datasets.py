import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import pickle
from gpflow.config import default_float

rotMNIST_train_path = '../data/mnist_rotation/mnist_all_rotation_normalized_float_train_valid.amat'
rotMNIST_test_path = '../data/mnist_rotation/mnist_all_rotation_normalized_float_test.amat'


def load(args):
    if args.dataset == 'MNIST':
        if args.subset_size is not None:
            if args.subset_size == 1250:
                training_ds, train_info = tfds.load('mnist', split=tfds.core.ReadInstruction('train', from_ = 1500, to = 1500 + args.subset_size, unit = 'abs'), with_info=True) # , shuffle_files=True
            else:
                training_ds, train_info = tfds.load('mnist', split=tfds.core.ReadInstruction('train', from_ = 0, to = args.subset_size, unit = 'abs'), with_info=True) # , shuffle_files=True)
        else: 
            training_ds, train_info = tfds.load('mnist', split='train', with_info=True) 
        test_ds, test_info = tfds.load(name="mnist", split='test', with_info=True)
        training_ds = training_ds.map(lambda x: (tf.cast(x['image'], tf.float32) / 255., x['label']))
        test_ds = test_ds.map(lambda x: (tf.cast(x['image'], tf.float32) / 255., x['label']))

        return training_ds, test_ds

    if args.dataset == 'rotMNIST':
        train_data = np.loadtxt(rotMNIST_train_path)
        test_data = np.loadtxt(rotMNIST_test_path)

        if args.subset_size is not None:
            nr_train_datapoints = args.subset_size
            nr_test_datapoints = 500  # test on small subset also for toy experiments

        else:
            nr_train_datapoints = train_data.shape[0]
            nr_test_datapoints = test_data.shape[0]

        train_images = train_data[:nr_train_datapoints, :-1]
        train_images = train_images.reshape([nr_train_datapoints, 28, 28, 1], order='F')
        train_labels = train_data[:nr_train_datapoints, -1]

        test_images = test_data[:nr_test_datapoints, :-1]
        test_images = test_images.reshape([-1, 28, 28, 1], order='F')
        test_labels = test_data[:nr_test_datapoints, -1]

        train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
        test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))

        return train_dataset, test_dataset


    if args.dataset == 'CIFAR10':
        if args.subset_size is not None:
            training_ds, train_info = tfds.load('cifar10', split=tfds.core.ReadInstruction('train', from_=0, to=args.subset_size, unit='abs'), with_info=True) # , shuffle_files=True)
        else:
            training_ds, train_info = tfds.load("cifar10", split='train', with_info=True)  # , shuffle_files=True)
        test_ds, test_info = tfds.load(name="cifar10", split='test', with_info=True)
        training_ds = training_ds.map(lambda x: (tf.cast(x['image'], tf.float32) / 255., x['label']))
        test_ds = test_ds.map(lambda x: (tf.cast(x['image'], tf.float32) / 255., x['label']))

        return training_ds, test_ds

    if args.dataset == 'PCAM':  # full PCAM, 96 x 96 pixels
        training_ds, train_info = tfds.load('patch_camelyon', split=tfds.core.ReadInstruction('train', from_ = 0, to = args.subset_size, unit = 'abs'), with_info=True) # , shuffle_files=True)
        training_ds = training_ds.map(lambda x: (tf.cast(x['image'], tf.float32) / 255., x['label']))

        test_ds, test_info = tfds.load(name="patch_camelyon", split='test', with_info=True)
        test_ds = test_ds.map(lambda x: (tf.cast(x['image'], tf.float32) / 255., x['label']))

        return training_ds, test_ds

    if args.dataset == 'PCAM32':  # PCAM, 32 x 32 pixels
        training_ds, train_info = tfds.load('patch_camelyon', split=tfds.core.ReadInstruction('train', from_ = 0, to = args.subset_size, unit = 'abs'), with_info=True) # , shuffle_files=True)
        training_ds = training_ds.map(lambda x: (tf.cast(x['image'], tf.float32)[32:64,32:64,:] / 255., x['label']))

        test_ds, test_info = tfds.load(name="patch_camelyon", split='validation', with_info=True)
        test_ds = test_ds.map(lambda x: (tf.cast(x['image'], tf.float32)[32:64,32:64,:] / 255., x['label']))

        return training_ds, test_ds


def create_full_batch(dataset, nr_datapoints=50000, return_labels=False):
    length = 0
    all_data = []
    all_labels = []
    for batch in dataset:
        all_data.append(batch[0])
        all_labels.append(batch[1])
        length += len(batch)
        if length > nr_datapoints:
            break
    all_data = tf.experimental.numpy.vstack([batch for batch in all_data])
    all_labels = tf.experimental.numpy.vstack([batch for batch in all_labels])
    if return_labels:
        return all_data, all_labels
    return all_data
