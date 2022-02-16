import tensorflow as tf
from deepkernelinv_experiments.model_maker import create_model, create_optimizer, create_map_fn, initialize_inducing_points, make_evaluation_step, CNN
from deepkernelinv_experiments.utils import load_datasets, helpers
import gpflow
from gpflow.config import default_float
from tensorflow.keras import backend


# tf.config.experimental_run_functions_eagerly(True)
def plot_step(model, train_dataset):
    import matplotlib.pyplot as plt
    test_data_iterator = iter(train_dataset)
    for batch in test_data_iterator:
        images = batch[0][:5]
        reconstructions = model(images)
        for img in range(5):
            plt.imshow(images[img])
            plt.savefig('reconstructions/img_%s.png' %img)
            plt.imshow(reconstructions[img])
            plt.savefig('reconstructions/rec_%s.png' %img)


def test_step(args, model, test_dataset, nr_batches=None):
    test_data_iterator = iter(test_dataset)
    test_acc = tf.zeros((), dtype=default_float())
    batch_ix = tf.zeros((), dtype=default_float())
    for batch in test_data_iterator:
        batch_acc = make_evaluation_step(args, batch, model)
        test_acc += batch_acc
        batch_ix += 1
        if batch_ix == nr_batches:
            break
    test_acc /= batch_ix
    return test_acc


def compute_full_batch_elbo(args, model, train_dataset, nr_batches=None):
    print('computing full batch elbo')
    train_data_iterator = iter(train_dataset)
    total_elbo = tf.zeros((), dtype=default_float())
    batch_ix = tf.zeros((), dtype=default_float())
    for batch in train_data_iterator:
        batch_elbo = model.elbo(batch)
        total_elbo += batch_elbo
        batch_ix += 1
        if batch_ix == nr_batches:
            break
    total_elbo /= batch_ix
    return total_elbo


def test(args):
    training_ds, test_ds = load_datasets.load(args)
    map_fn = create_map_fn(args)

    train_dataset = training_ds \
        .shuffle(1024) \
        .batch(args.batch_size[0], drop_remainder=True) \
        .map(map_fn, num_parallel_calls=args.autotune) \
        .prefetch(args.autotune)
    test_dataset = test_ds\
        .batch(args.batch_size[0], drop_remainder=True)\
        .map(map_fn, num_parallel_calls=args.autotune) \
        .prefetch(args.autotune)

    #if 'GP' in args.use_model:
        # these need to be initialized in order to initialize the model but will be overwritten by restoring from ckpt
    #    args.initial_inducing_points = initialize_inducing_points(args, None, method='zeros')

    model = create_model(args, train_dataset)
    print('before loading')
    gpflow.utilities.print_summary(model)

    optimizer = create_optimizer(args)
    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer, step=tf.Variable(0))
    print('LOADING FROM PATH', args.model_path)
    latest_checkpoint = tf.train.latest_checkpoint(args.model_path)
    if latest_checkpoint is None:
        print('NO CHECKPOINT TO LOAD FROM - CHECK MODEL!')
        exit()
    print('RELOADING FROM', latest_checkpoint)
    ckpt.restore(latest_checkpoint)

    print('after loading')
    gpflow.utilities.print_summary(model)

    print('STEP IS', ckpt.step)

    test_acc = test_step(args, model, test_dataset)
    print('Final test acc. is', test_acc)

    return model, train_dataset, test_dataset
