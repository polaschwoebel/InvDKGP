import tensorflow as tf
from utils import load_datasets
from deepkernelinv_experiments.utils import toggle_learnable_params
from datetime import datetime
import utils 
from model_maker import create_model, create_optimizer, create_map_fn
from model_maker.CNN import load_pretrained_cnn

import gpflow
from gpflow.utilities import set_trainable
from deepkernelinv_experiments.utils.set_trainable_params import set_trainable_basekern_cnn

from .train import make_logger, run_gp_optimization


def train(args):
    # always start with first value (GP training first or there's only one)
    batch_size = args.batch_size[0]

    start_time = datetime.now()
    training_ds, test_ds = load_datasets.load(args)
    map_fn = create_map_fn(args)

    train_dataset = training_ds \
        .shuffle(1024) \
        .batch(batch_size, drop_remainder=False) \
        .map(map_fn, num_parallel_calls=args.autotune) \
        .prefetch(args.autotune)
    test_dataset = test_ds\
        .batch(batch_size, drop_remainder=False)\
        .map(map_fn, num_parallel_calls=args.autotune) \
        .prefetch(args.autotune)

    model = create_model(args, train_dataset)
    optimizer = create_optimizer(args)

    if args.fix_indp:
        set_trainable(model.inducing_variable.Z, False)

    print('FINAL MODEL')
    gpflow.utilities.print_summary(model)

    step = tf.Variable(0, dtype=tf.int32, trainable=False)
    ckpt = tf.train.Checkpoint(model=model, step=step, optimizer=optimizer)
    manager = tf.train.CheckpointManager(ckpt, args.model_path, max_to_keep=5)

    # tensorboard logging
    monitors = make_logger(args, model, train_dataset, test_dataset)

    # continue training
    if args.continue_training:
        if len(args.batch_size) > 1:
            raise NotImplementedError  # not implemented yet for mixed bs/lr/os
        else:
            print('LOADING FROM PATH', args.model_path)
            latest_checkpoint = tf.train.latest_checkpoint(args.model_path)
            if latest_checkpoint is None:
                print('NO CHECKPOINT TO LOAD FROM - CHECK MODEL!')
                exit()
            print('RELOADING FROM', latest_checkpoint)
            ckpt.restore(latest_checkpoint)
            print('STEP IS', ckpt.step)
            print('OPTIMIZER', ckpt.optimizer)
            # print('LOADED MODEL')
            # gpflow.utilities.print_summary(model)
            optimizer = ckpt.optimizer

    # run optimization from scratch
    total_nr_iterations = args.nr_epochs * (args.total_nr_data // batch_size)
    print('Running optimization for %s steps.' % total_nr_iterations)
    print('Total number of datapoints is', args.total_nr_data)
    print('experiment config is', args.experiment_config)

    # joint training loop
    if args.experiment_config == 'vanilla':
        print('Running regular optimization loop for %s steps.' % total_nr_iterations)
        run_gp_optimization(
            args, model, total_nr_iterations, train_dataset, test_dataset,
            manager, ckpt, monitors, optimizer, start_time)

    # coordinate ascent training loop
    elif args.experiment_config == 'coord_ascent_training':
        print('Running coordinate ascent training loop.')
        # start by training the GP
        set_trainable_basekern_cnn(args, model, False) 

        while ckpt.step < total_nr_iterations:
            print('optimizer is', optimizer)
            print('learning_rate is', optimizer.learning_rate)
            for one_batch in train_dataset:
                print('batch size is', one_batch[0].shape[0])
                break
            print('Model summary after toggling:')
            gpflow.utilities.print_summary(model)

            run_gp_optimization(
              args, model, args.toggle_after_steps, train_dataset, test_dataset,
              manager, ckpt, monitors, optimizer, start_time)
            utils.toggle_learnable_params(args, model)

            train_dataset, test_dataset, batch_size = utils.toggle_orbit_batch_size(
                args, batch_size, map_fn, training_ds, test_ds, model) # only return new datasets here, the rest is done in-place
            optimizer = create_optimizer(args)
