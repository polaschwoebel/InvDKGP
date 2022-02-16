from model_maker import create_model, create_map_fn, CNN
from .test import test_step
import gpflow
import tensorflow as tf
from gpflow.monitor import (
    Monitor,
    MonitorTaskGroup,
    ScalarToTensorBoard,
)
from utils import CustomModelToTensorBoard, load_datasets
import numpy as np
from utils.data_augmentation import (
    augment_color, augment_affine, augment_mirror_translate_zeropad, augment_mirror_translate_nearestpad,
    augment_translate_zeropad, augment_translate_nearestpad, augment_rotate, augment_mirror, augment_rotate_orbit, augment_rot_quant_orbit
)
from invgp.kernels import orbits


def make_logger(args, model, train_dataset, test_dataset):
    model_task = CustomModelToTensorBoard(args.tensorboard_logdir, model)
    ll_task = ScalarToTensorBoard(
        args.tensorboard_logdir, lambda loss, learning_rate: -loss, "log_likelihood")
    train_acc_task = ScalarToTensorBoard(
        args.tensorboard_logdir,
        lambda loss, learning_rate: test_step(args, model, train_dataset), "train_acc")
    test_acc_task = ScalarToTensorBoard(
        args.tensorboard_logdir,
        lambda loss, learning_rate: test_step(args, model, test_dataset), "test_acc")
    frequent_logging_taks = MonitorTaskGroup(
        [model_task, ll_task], period=args.minibatch_frequency)
    infrequent_logging_taks = MonitorTaskGroup(
        [train_acc_task, test_acc_task], period=args.full_batch_frequency)
    lr_task = ScalarToTensorBoard(
        args.tensorboard_logdir, 
        lambda loss, learning_rate: learning_rate, "learning_rate")
    monitor = Monitor(MonitorTaskGroup([frequent_logging_taks, infrequent_logging_taks, lr_task]))

    return monitor


def optimization_step(batch, model, step, monitor):
    batch_loss, batch_acc = model.train_on_batch(*batch[:-1], batch[-1], reset_metrics=True)
    monitor(step, loss=batch_loss, learning_rate=model.optimizer._decayed_lr(tf.float32))
    return batch_loss, batch_acc


def run_cnn_optimization(args, model, iterations, train_dataset, test_dataset,
                         manager, ckpt, monitor):
    train_iter = iter(train_dataset.repeat())
    # print('trainable params:', np.sum([np.prod(v.get_shape().as_list()) for v in model.trainable_variables]))

    progbar = tf.keras.utils.Progbar(iterations+1)
    for step in tf.range(iterations + 1):
        batch = next(train_iter)
        batch_loss, batch_acc = optimization_step(batch, model, step, monitor)

        if step % args.full_batch_frequency == 10:
            ckpt_path = manager.save()
            print("\nSaved checkpoint at: %s" % ckpt_path)
        ckpt.step.assign_add(1)
        progbar.update(step.numpy()+1)

    ckpt_path = manager.save()


def train(args):
    color_orbit = orbits.ColorTransform(minibatch_size=args.batch_size[0], log_lims_contrast=args.DA_params_contrast, log_lims_brightness=args.DA_params_brightness)
    affine_orbit = orbits.InterpretableSpatialTransform(
        minibatch_size=args.batch_size[0],
        theta_min=np.array([0., 1., 1., 0., 0., 0., 0.]) - args.DA_params_affine * np.array([3.14, 1., 1., 1., 1., 1., 1.]), 
        theta_max=np.array([0., 1., 1., 0., 0., 0., 0.]) + args.DA_params_affine * np.array([3.14, 1., 1., 1., 1., 1., 1.]), 
        colour=True, radians=True)
    pcam_affine_orbit = orbits.InterpretableSpatialTransform(
        minibatch_size=args.batch_size[0],  # r, scale x, scale y, sheer x, sheer y, translate x, translate y
        theta_min=np.array([3.14, 1., 1., 0., 0., 0.0625, 0.0625]),
        theta_max=np.array([-3.14, 1., 1., 0., 0., -0.0625, -0.0625]), 
        colour=True, radians=True)
    pcam_rot_orbit = orbits.ImageRotation(
            angle=3.1 * 0.1,  # hardcoded for now 
            radians=True,  # hardcoded for now 
            input_dim=tf.cast(args.image_size, np.float32),
            minibatch_size=1,
            use_stn=True,
            interpolation_method="BILINEAR")
    pcam_rot_quant_orbit = orbits.ImageRotQuant(
            orbit_size=2,  # always start with the first one (GP training)
            angle=180.,
            input_dim=args.image_size,
            interpolation_method="BILINEAR", 
            use_stn=True)

    if args.dataset in ['PCAM', 'PCAM32']:
        affine_orbit = pcam_affine_orbit

    def rotator(image, label):
        if args.DA_mirror_translate_rotate or args.DA_mirror_rotate:
            image, label = augment_rotate(image, label)
        else:
            image, label = image, label
        return (image, label)

    # for standard DA 
    def augmentor(image, label):
        if args.DA_color:
            data = (image, label)
            image, label = augment_color(
                data, color_orbit)

        if args.DA_rotate:
            data = (image, label)
            image, label = augment_rotate_orbit(
                data, pcam_rot_orbit)

        if args.DA_rot_quant:
            data = (image, label)
            image, label = augment_rot_quant_orbit(
                data, pcam_rot_quant_orbit)

        if args.DA_affine:
            data = (image, label)
            image, label = augment_affine(
                data, affine_orbit)

        if args.DA_translate:
            if args.DA_pad == 'zeropad':
                image, label = augment_translate_zeropad(image, label)
            if args.DA_pad == 'nearestpad':
                image, label = augment_translate_nearestpad(image, label)

        if args.DA_mirror_rotate:
            image, label = augment_mirror(image, label)

        if args.DA_mirror_translate or args.DA_mirror_translate_rotate:  # already rotated 
            if args.DA_pad == 'zeropad':
                image, label = augment_mirror_translate_zeropad(image, label)
            if args.DA_pad == 'nearestpad':
                image, label = augment_mirror_translate_nearestpad(image, label)
        return (image, label)

    training_ds, test_ds = load_datasets.load(args)

    map_fn = create_map_fn(args)

    train_dataset = training_ds \
        .shuffle(1024) \
        .map(map_fn, num_parallel_calls=args.autotune) \
        .map(rotator, num_parallel_calls=args.autotune) \
        .batch(args.batch_size[0], drop_remainder=True) \
        .map(augmentor, num_parallel_calls=args.autotune) \
        .prefetch(args.autotune)
    test_dataset = test_ds \
        .batch(args.batch_size[0], drop_remainder=True)\
        .map(map_fn, num_parallel_calls=args.autotune) \
        .map(augmentor, num_parallel_calls=args.autotune)  # comment out when we  don't need "DA"/STN preprocessing also at test time 

    model = create_model(args)
    step = tf.Variable(1, dtype=tf.int32, trainable=False)
    ckpt = tf.train.Checkpoint(model=model, step=step)
    manager = tf.train.CheckpointManager(ckpt, args.model_path, max_to_keep=5)
    monitor = make_logger(args, model, train_dataset, test_dataset)

    if args.continue_training:
        print('LOADING FROM LATEST CHECKPOINT, PATH', args.model_path)
        latest_checkpoint = tf.train.latest_checkpoint(args.model_path)
        if latest_checkpoint is None:
            print('NO CHECKPOINT TO LOAD FROM - CHECK MODEL!')
            exit()
        print('RELOADING FROM', latest_checkpoint)
        ckpt.restore(latest_checkpoint)
        step = ckpt.step

    print('FINAL MODEL')
    gpflow.utilities.print_summary(model)

    # run optimization
    total_nr_iterations = args.nr_epochs * (args.total_nr_data // args.batch_size[0])
    print('Running optimization for %s steps.' % total_nr_iterations)

    run_cnn_optimization(
        args, model, total_nr_iterations, train_dataset, test_dataset,
        manager, ckpt, monitor)
