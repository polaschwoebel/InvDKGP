from gpflow.monitor import (
    Monitor,
    MonitorTaskGroup,
    ScalarToTensorBoard,
)
from .test import test_step, compute_full_batch_elbo
from utils import CustomModelToTensorBoard
import tensorflow as tf
from datetime import datetime


def make_logger(args, model, train_dataset, test_dataset):
    model_task = CustomModelToTensorBoard(args.tensorboard_logdir, model)
    minibatch_elbo_task = ScalarToTensorBoard(
        args.tensorboard_logdir, lambda loss, learning_rate: -loss, "elbo")
    full_batch_elbo_task = ScalarToTensorBoard(
        args.tensorboard_logdir, lambda duration_per_step: compute_full_batch_elbo(args, model, train_dataset, nr_batches=args.nr_eval_batches), "full_batch_elbo" )
    train_acc_task = ScalarToTensorBoard(
        args.tensorboard_logdir,
        lambda duration_per_step: test_step(args, model, train_dataset, nr_batches=args.nr_eval_batches), "train_acc")
    test_acc_task = ScalarToTensorBoard(
        args.tensorboard_logdir,
        lambda duration_per_step: test_step(args, model, test_dataset, nr_batches=args.nr_eval_batches), "test_acc")
    timing_task = ScalarToTensorBoard(
        args.tensorboard_logdir,
        lambda duration_per_step: duration_per_step, "seconds_per_step")
    lr_task = ScalarToTensorBoard(
        args.tensorboard_logdir, 
        lambda loss, learning_rate: learning_rate, "learning_rate")
    frequent_logging_taks = MonitorTaskGroup(
        [model_task, minibatch_elbo_task, lr_task], period=args.minibatch_frequency)
    infrequent_logging_taks = MonitorTaskGroup(
        [train_acc_task, test_acc_task, full_batch_elbo_task, timing_task])
    frequent_monitor = Monitor(frequent_logging_taks)
    infrequent_monitor = Monitor(infrequent_logging_taks)
    return frequent_monitor, infrequent_monitor


def run_gp_optimization(args, model, iterations, train_dataset, test_dataset,
                        manager, ckpt, monitors, optimizer, start_time):
    frequent_monitor, infrequent_monitor = monitors
    progbar = tf.keras.utils.Progbar(iterations+1)
    train_iter = iter(train_dataset.repeat())

    @tf.function(autograph=True)
    def optimization_step(batch, step):
        with tf.GradientTape() as tape:
            tape.watch(model.trainable_variables)
            loss = -model.elbo(batch)
            grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        frequent_monitor(step, loss=loss, learning_rate=optimizer._decayed_lr(tf.float32)) 

    for i in tf.range(iterations + 1):
        if ckpt.step % args.full_batch_frequency == 0:  # save model every 1 + x steps
            ckpt_path = manager.save()
            print("\nSaved checkpoint at: %s" % ckpt_path)
            duration_timedelta = datetime.now() - start_time
            duration = duration_timedelta.total_seconds()
            if ckpt.step == 2: 
                duration_per_step = duration / 2
            else:
                duration_per_step = duration / args.full_batch_frequency
            print('Duration:', str(duration_timedelta).split(".")[0])  # remove microseconds for display
            infrequent_monitor(ckpt.step, duration_per_step=duration_per_step)
            # reset start time 
            start_time = datetime.now()
        
        batch = next(train_iter)
        optimization_step(batch, ckpt.step)
        ckpt.step.assign_add(1)
        progbar.update(i.numpy()+1)
    # always save the model in the end
    ckpt_path = manager.save()
