import tensorflow as tf
from . import create_optimizer
from gpflow.config import default_float
import gpflow
from .DA_layers import DALayer, AggregatePredictionsLayer
from deepkernelinv_experiments.utils.helpers import name_mapper
from .LieConv import MolecLieResNet

def initialize_cnn(args):
    image_shape = args.image_shape
    input_size = int(tf.reduce_prod(image_shape))

    if args.CNN_architecture == 'MLE_learned_da': # for now we handle the MLE+DA experiment like this, not the nicest
        small_net = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=image_shape, batch_size=args.batch_size[0]),
            DALayer(args),
            tf.keras.layers.Conv2D(filters=20, padding="same", activation="relu", input_shape=image_shape, kernel_size=(5, 5)),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2),
            tf.keras.layers.Conv2D(filters=50, kernel_size=(5, 5), padding="same", activation="relu"),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(500, activation="relu"),
            tf.keras.layers.Dense(args.latent_dim, activation=None),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(args.nr_classes, activation='softmax'),
            AggregatePredictionsLayer(args),
            tf.keras.layers.Lambda(lambda x: tf.cast(x,  default_float()))])
        return small_net

    if args.CNN_architecture == 'small':
        small_net = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=20, padding="same", activation="relu", input_shape=image_shape, kernel_size=(5, 5)),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2),
            tf.keras.layers.Conv2D(filters=50, kernel_size=(5, 5), padding="same", activation="relu"),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(500, activation="relu"),
            tf.keras.layers.Dense(args.latent_dim, activation=None),
            # tf.keras.layers.BatchNormalization(trainable=True), 
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(args.nr_classes, activation='softmax'),
            tf.keras.layers.Lambda(lambda x: tf.cast(x,  default_float()))])
        return small_net

    if args.CNN_architecture == 'large':
        large_net = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=image_shape),
            tf.keras.layers.GaussianNoise(0.15),
            tf.keras.layers.Conv2D(filters=128, padding="same", strides=1, input_shape=image_shape, kernel_size=(3, 3)),
            #tf.keras.layers.BatchNormalization(trainable=True), 
            tf.keras.layers.LeakyReLU(alpha=0.1),
            tf.keras.layers.Conv2D(filters=128, padding="same", strides=1, kernel_size=(3, 3)),
            #tf.keras.layers.BatchNormalization(trainable=True), 
            tf.keras.layers.LeakyReLU(alpha=0.1),
            tf.keras.layers.Conv2D(filters=128, padding="same", strides=1, kernel_size=(3, 3)),
            #tf.keras.layers.BatchNormalization(trainable=True), 
            tf.keras.layers.LeakyReLU(alpha=0.1),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2),
            tf.keras.layers.Dropout(.5) if args.dropout else tf.keras.Sequential(), 
            tf.keras.layers.Conv2D(filters=256, padding="same", strides=1, kernel_size=(3, 3)),
            #tf.keras.layers.BatchNormalization(trainable=True), 
            tf.keras.layers.LeakyReLU(alpha=0.1),
            tf.keras.layers.Conv2D(filters=256, padding="same", strides=1, kernel_size=(3, 3)),
            #tf.keras.layers.BatchNormalization(trainable=True), 
            tf.keras.layers.LeakyReLU(alpha=0.1),
            tf.keras.layers.Conv2D(filters=256, padding="same", strides=1, kernel_size=(3, 3)),
            #tf.keras.layers.BatchNormalization(trainable=True),            
            tf.keras.layers.LeakyReLU(alpha=0.1),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2),
            tf.keras.layers.Dropout(.3) if args.dropout else tf.keras.Sequential(),
            tf.keras.layers.Conv2D(filters=512, padding="same", strides=1, kernel_size=(3, 3)),
            #tf.keras.layers.BatchNormalization(trainable=True), 
            tf.keras.layers.LeakyReLU(alpha=0.1),
            tf.keras.layers.Conv2D(filters=256, padding="same", strides=1, kernel_size=(3, 3)),
            #tf.keras.layers.BatchNormalization(trainable=True), 
            tf.keras.layers.LeakyReLU(alpha=0.1),
            tf.keras.layers.Conv2D(filters=50, padding="valid", strides=1, kernel_size=(3, 3)), # 128
            #tf.keras.layers.BatchNormalization(trainable=True), 
            tf.keras.layers.LeakyReLU(alpha=0.1),
            tf.keras.layers.GlobalAveragePooling2D(data_format='channels_last'),
            tf.keras.layers.Dense(args.nr_classes, activation='softmax'),
            tf.keras.layers.Lambda(lambda x: tf.cast(x,  default_float()))])
        return large_net
        

    if args.CNN_architecture == 'litjens_vgg':
        padding = 'same' if args.dataset == 'PCAM32' else 'valid' # use padding when data is tiny
        vgg = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=image_shape),
            # 1. conv blovk
            tf.keras.layers.Conv2D(filters=16, padding=padding, strides=1, input_shape=image_shape, kernel_size=(3, 3)),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(filters=16, padding=padding, strides=1, input_shape=image_shape, kernel_size=(3, 3)),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2),
            # 2. conv block
            tf.keras.layers.Conv2D(filters=32, padding=padding, strides=1, input_shape=image_shape, kernel_size=(3, 3)),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(filters=32, padding=padding, strides=1, input_shape=image_shape, kernel_size=(3, 3)),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2),
            # 3. conv block
            tf.keras.layers.Conv2D(filters=64, padding=padding, strides=1, input_shape=image_shape, kernel_size=(3, 3)),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(filters=64, padding=padding, strides=1, input_shape=image_shape, kernel_size=(3, 3)),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2),
            # FC top
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dropout(args.dropout_prob),  # if args.dropout else tf.keras.Sequential(), 
            tf.keras.layers.Dense(args.latent_dim, activation=None),
            tf.keras.layers.Dropout(args.dropout_prob),  # if args.dropout else tf.keras.Sequential(), 
            tf.keras.layers.Dense(args.nr_classes, activation='softmax'),
            tf.keras.layers.Lambda(lambda x: tf.cast(x,  default_float()))])
        print(vgg.summary())
        return vgg

    if args.CNN_architecture == 'lie_conv':
        num_species = 6 #5 5 in original code but shouldnt it be 6? 
        charge_scale = 9 # not in use currently 
        input_shape = (29, 6)
        lie_conv_model = MolecLieResNet(charge_scale=charge_scale, bn=False, batch_size=args.batch_size) # TODO: implement bn
        # lie_conv_model.build(input_shape=input_shape)
        print(lie_conv_model.summary())
        return lie_conv_model


# similar to the "regular" CNN but without the last fully connected layer
def initialize_deep_kernel_cnn(args):
    image_shape = args.image_shape
    input_size = int(tf.reduce_prod(image_shape))

    if args.CNN_architecture == 'MLE_learned_da': 
        print('MLE + DKL does not make sense.')
        exit()

    if args.CNN_architecture == 'small':
        small_net = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=20, padding="same", activation="relu", input_shape=image_shape, kernel_size=(5, 5)),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2),
            tf.keras.layers.Conv2D(filters=50, kernel_size=(5, 5), padding="same", activation="relu"),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(500, activation="relu"),
            tf.keras.layers.Dense(args.latent_dim, activation=None),
            # tf.keras.layers.BatchNormalization(trainable=True),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Lambda(lambda x: tf.cast(x,  default_float()))])
        return small_net

    if args.CNN_architecture == 'large':
        large_net = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=image_shape),
            tf.keras.layers.GaussianNoise(0.15),
            tf.keras.layers.Conv2D(filters=128, padding="same", strides=1, input_shape=image_shape, kernel_size=(3, 3)),
            #tf.keras.layers.BatchNormalization(trainable=True), 
            tf.keras.layers.LeakyReLU(alpha=0.1),
            tf.keras.layers.Conv2D(filters=128, padding="same", strides=1, kernel_size=(3, 3)),
            #tf.keras.layers.BatchNormalization(trainable=True), 
            tf.keras.layers.LeakyReLU(alpha=0.1),
            tf.keras.layers.Conv2D(filters=128, padding="same", strides=1, kernel_size=(3, 3)),
            #tf.keras.layers.BatchNormalization(trainable=True), 
            tf.keras.layers.LeakyReLU(alpha=0.1),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2),
            tf.keras.layers.Dropout(.5) if args.dropout else tf.keras.Sequential(), 
            tf.keras.layers.Conv2D(filters=256, padding="same", strides=1, kernel_size=(3, 3)),
            #tf.keras.layers.BatchNormalization(trainable=True), 
            tf.keras.layers.LeakyReLU(alpha=0.1),
            tf.keras.layers.Conv2D(filters=256, padding="same", strides=1, kernel_size=(3, 3)),
            #tf.keras.layers.BatchNormalization(trainable=True), 
            tf.keras.layers.LeakyReLU(alpha=0.1),
            tf.keras.layers.Conv2D(filters=256, padding="same", strides=1, kernel_size=(3, 3)),
            #tf.keras.layers.BatchNormalization(trainable=True),            
            tf.keras.layers.LeakyReLU(alpha=0.1),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2),
            tf.keras.layers.Dropout(.3) if args.dropout else tf.keras.Sequential(),
            tf.keras.layers.Conv2D(filters=512, padding="same", strides=1, kernel_size=(3, 3)),
            #tf.keras.layers.BatchNormalization(trainable=True), 
            tf.keras.layers.LeakyReLU(alpha=0.1),
            tf.keras.layers.Conv2D(filters=256, padding="same", strides=1, kernel_size=(3, 3)),
            #tf.keras.layers.BatchNormalization(trainable=True), 
            tf.keras.layers.LeakyReLU(alpha=0.1),
            tf.keras.layers.Conv2D(filters=50, padding="valid", strides=1, kernel_size=(3, 3)), # 128
            #tf.keras.layers.BatchNormalization(trainable=True), 
            tf.keras.layers.LeakyReLU(alpha=0.1),
            tf.keras.layers.GlobalAveragePooling2D(data_format='channels_last'),
            tf.keras.layers.Lambda(lambda x: tf.cast(x,  default_float()))])
        return large_net

    if args.CNN_architecture == 'litjens_vgg':
        padding = 'same' if args.dataset == 'PCAM32' else 'valid' # use padding when data is tiny
        vgg = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=image_shape),
            # 1. conv blovk
            tf.keras.layers.Conv2D(filters=16, padding=padding, strides=1, input_shape=image_shape, kernel_size=(3, 3)),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(filters=16, padding=padding, strides=1, input_shape=image_shape, kernel_size=(3, 3)),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2),
            # 2. conv block
            tf.keras.layers.Conv2D(filters=32, padding=padding, strides=1, input_shape=image_shape, kernel_size=(3, 3)),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(filters=32, padding=padding, strides=1, input_shape=image_shape, kernel_size=(3, 3)),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2),
            # 3. conv block
            tf.keras.layers.Conv2D(filters=64, padding=padding, strides=1, input_shape=image_shape, kernel_size=(3, 3)),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(filters=64, padding=padding, strides=1, input_shape=image_shape, kernel_size=(3, 3)),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2),
            # FC top
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dropout(args.dropout_prob),  # if args.dropout else tf.keras.Sequential(), 
            tf.keras.layers.Dense(args.latent_dim, activation=None),  # args.latent_dim = 128
            tf.keras.layers.Lambda(lambda x: tf.cast(x,  default_float()))])
        print(vgg.summary())
        return vgg


def load_pretrained_cnn(args):
    print('Intializing with pretrained CNN.')
    # initialize CNN and load parameters
    pretrained_CNN = initialize_cnn(args)
    deep_kernel_CNN = initialize_deep_kernel_cnn(args)
    ckpt = tf.train.Checkpoint(model=pretrained_CNN)
    #if args.checkpoint_path is not None: # load from specific checkpoint
    #        latest_checkpoint = tf.train.latest_checkpoint(args.checkpoint_path)
   # else:  # load from filepath
    print('LOADING FROM LATEST CHECKPOINT, PATH', args.pretrained_CNN_path)
    latest_checkpoint = tf.train.latest_checkpoint(args.pretrained_CNN_path)

    ckpt.restore(latest_checkpoint).expect_partial()
    if latest_checkpoint is None:
        print('NO CHECKPOINT TO LOAD FROM - CHECK PATH!')
        exit()
    print('LOADING CNN WEIGHTS FROM', latest_checkpoint)
    # print('PRETRAINED CNN')
    # gpflow.utilities.print_summary(pretrained_CNN)
    pretrained_CNN_param_dict = gpflow.utilities.parameter_dict(pretrained_CNN)
    target_params = gpflow.utilities.parameter_dict(deep_kernel_CNN).keys()

    params_to_load = {key: value for (key, value) in pretrained_CNN_param_dict.items() if key in target_params}
    # params_to_load = {key: tf.cast(value, default_float()) for (key, value) in pretrained_CNN_param_dict.items() if key in target_params}
    gpflow.utilities.multiple_assign(deep_kernel_CNN, params_to_load)
    # print('LOADED INTO NEW CNN')
    # gpflow.utilities.print_summary(deep_kernel_CNN)
    return deep_kernel_CNN


def load_pretrained_encoder(args):
    print('Loading pretrained autoencoder')
    #autoencoder = tf.keras.load_model(args.pretrained_CNN_path)
    from experiment_scripts.AE import create_model as initialize_ae
    ae = initialize_ae(args)
    print('ae initialized')
    deep_kernel_CNN = initialize_deep_kernel_cnn(args)
    ckpt = tf.train.Checkpoint(model=ae)
    latest_checkpoint = tf.train.latest_checkpoint(args.pretrained_CNN_path)
    ckpt.restore(latest_checkpoint).expect_partial()
    if latest_checkpoint is None:
        print('NO CHECKPOINT TO LOAD FROM - CHECK PATH!')
        exit()
    print('LOADING CNN WEIGHTS FROM', latest_checkpoint)
    pretrained_CNN_param_dict = gpflow.utilities.parameter_dict(ae)
    pretrained_CNN_param_dict = {key[11:]: value for (key, value) in pretrained_CNN_param_dict.items() if key.startswith('._layers[1]')}
    target_params = gpflow.utilities.parameter_dict(deep_kernel_CNN).keys()
    params_to_load = {key: value for (key, value) in pretrained_CNN_param_dict.items() if key in target_params}
    gpflow.utilities.multiple_assign(deep_kernel_CNN, params_to_load)
    return deep_kernel_CNN


def load_pretrained_CNN_MLE_DA(args):
    # CNN_model.compile(loss='categorical_crossentropy', metrics=['accuracy'])
    import copy 
    cnn_args = copy.deepcopy(args)
    cnn_args.CNN_architecture = 'small'
    pretrained_CNN = initialize_cnn(cnn_args)
    new_CNN = initialize_cnn(args)
    print('NEW CNN')
    gpflow.utilities.print_summary(new_CNN)

    ckpt = tf.train.Checkpoint(model=pretrained_CNN)
    latest_checkpoint = tf.train.latest_checkpoint(args.pretrained_CNN_path)
    ckpt.restore(latest_checkpoint).expect_partial()
    if latest_checkpoint is None:
        print('NO CHECKPOINT TO LOAD FROM - CHECK PATH!')
        exit()
    print('LOADING CNN WEIGHTS FROM', latest_checkpoint)
    print('PRETRAINED CNN')
    gpflow.utilities.print_summary(pretrained_CNN)
    pretrained_CNN_param_dict = gpflow.utilities.parameter_dict(pretrained_CNN)
    params_to_load = {name_mapper(key): value for (key, value) in pretrained_CNN_param_dict.items()}
    # print('pretrained:', pretrained_CNN_param_dict.keys())
    # print('params to load:', params_to_load.keys())
    # #\print('target', target_params)
    gpflow.utilities.multiple_assign(new_CNN, params_to_load)
    # print('LOADED INTO NEW CNN')
    # gpflow.utilities.print_summary(new_CNN)
    # exit()
    return new_CNN


def qm9_loss(y_actual, y_predicted):
    print('y_predicted', y_predicted)
    print('y_actual', y_actual)
    return (y_predicted-y_actual).abs().mean()


def create_model(args, X=None):
    CNN_model = initialize_cnn(args)

    if args.CNN_architecture == "MLE_learned_da" and args.pretrained_CNN_path is not None:
        CNN_model = load_pretrained_CNN_MLE_DA(args)

    optimizer = create_optimizer(args)
    loss = [qm9_loss] if args.dataset == 'QM9' else ['categorical_crossentropy']
    metrics = [qm9_loss] if args.dataset == 'QM9' else ['accuracy']
    CNN_model.compile(
        loss=loss, optimizer=optimizer, metrics=metrics)
    return CNN_model


def evaluation_step(nr_classes, batch, model):
    batch_loss, batch_acc = model.test_on_batch(batch[0], batch[1], reset_metrics=True)
    return batch_acc
