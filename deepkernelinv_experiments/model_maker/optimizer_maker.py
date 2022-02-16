import tensorflow as tf
import gpflow
from gpflow.config import default_float


def create_optimizer(args, use_optimizer, learning_rate, decay_lr, iterations):
    if decay_lr == 'smooth':
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.002, decay_steps=15000,
            decay_rate=0.6, staircase=True)
    elif decay_lr == 'steps':
        boundaries = [int(iterations*0.5), int(iterations*0.75)]
        values = [learning_rate, learning_rate * 0.1, learning_rate * 0.01]
        lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            boundaries, values)
    elif decay_lr == 'cyclic':
        boundaries = [int(iterations*0.2), int(iterations*0.4), int(iterations*0.6), int(iterations*0.8)]
        values = [learning_rate * 0.01, learning_rate * 0.1, learning_rate, learning_rate * 0.1, learning_rate * 0.01]
        lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            boundaries, values)

    else:
        lr_schedule = learning_rate

    if use_optimizer == 'adam':
        beta_1 = 0.9
        beta_2 = 0.999
        print('optimizer: running with beta1/2', beta_1, beta_2)
        optimizer = tf.keras.optimizers.Adam(
                lr_schedule,
                beta_1=tf.Variable(beta_1, dtype=default_float()),
                beta_2=tf.Variable(beta_2, dtype=default_float()),
                epsilon=tf.Variable(1e-7, dtype=default_float()),
            )

    elif use_optimizer == 'sgd':
        optimizer = tf.optimizers.SGD(lr_schedule, momentum=0.9, nesterov=True)
    
    return optimizer
