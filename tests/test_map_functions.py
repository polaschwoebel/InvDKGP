from deepkernelinv_experiments.utils import load_datasets
from deepkernelinv_experiments.model_maker import create_map_fn
import numpy as np


class ObjectFromDict(object):
    def __init__(self, d):
        self.__dict__ = d


def test_map_functions():

    args_list = [
        {'use_model': model,  'dataset': dataset, 'likelihood': likelihood, 'STN_preprocessing': val, 'deepkernel': deepk}
            for model in ['CNN', 'SampleSVGP', 'SVGP']
            for dataset in ['MNIST', 'rotMNIST', 'CIFAR10'] 
            for likelihood in ['Gaussian', 'Softmax']
            for val in [True, False]
            for deepk in [True, False]
            ]

    for args in args_list:
        print('args is', args)
        args = ObjectFromDict(args)

        if args.use_model == 'CNN' and args.deepkernel:  # doesn't make sense 
            continue
        if args.dataset == 'CIFAR10' and args.STN_preprocessing:  # not implemented yet
            continue

        args.subset_size = None
        args.evaluate_on = 'test'

        map_fn = create_map_fn(args)
        training_ds, test_ds = load_datasets.load(args)

        train_dataset = training_ds \
            .shuffle(1024) \
            .batch(100, drop_remainder=False) \
            .map(map_fn)
        test_dataset = test_ds\
            .batch(100, drop_remainder=False)\
            .map(map_fn)

        X_train, Y_train = load_datasets.create_full_batch(train_dataset)
        X_test, Y_test = load_datasets.create_full_batch(test_dataset)

        print('shapes train:', X_train[0].shape, Y_train[0].shape)
        print('shapes test:', X_test[0].shape, Y_test[0].shape)

        print('values train:', X_train.min(), X_test.max())
        print('\n')

        # image shapes 
        if args.deepkernel or args.use_model == 'CNN':
            if 'MNIST' in args.dataset:
                assert X_train[0].shape == (28, 28, 1)
            if args.dataset == 'CIFAR10':
                assert X_train[0].shape == (32, 32, 3)    
        else:
            if 'MNIST' in args.dataset:
                assert X_train[0].shape == (784,)
            if args.dataset == 'CIFAR10':
                assert X_train[0].shape == (3072,)

        # label shape
        if args.likelihood == 'Gaussian' or args.use_model == 'CNN':
            assert Y_train[0].shape == (10,)
        elif args.likelihood == 'Softmax':
            assert Y_train[0].shape == (1,)

        # normalization
        np.testing.assert_allclose(X_train.min(), 0., atol=0.05)
        np.testing.assert_allclose(X_train.max(), 1., atol=0.05)


        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots(2)
        # ax[0].imshow(X_train[0][:, :, 0])
        # ax[0].set_title(Y_train[0])
        # ax[1].imshow(X_train[1][:, :, 0])
        # ax[1].set_title(Y_train[1])
        # plt.show()

        # exit()


if __name__ == '__main__':
    test_map_functions()