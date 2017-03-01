
import copy
import numpy as np
np.set_printoptions(suppress=True, precision=8)
import os
import sys
from sklearn.metrics import classification_report
import tensorflow as tf

path = os.path.join(os.path.dirname(__file__), os.pardir, 'neural_networks')
sys.path.append(os.path.abspath(path))
path = os.path.join(os.path.dirname(__file__), os.pardir)
sys.path.append(os.path.abspath(path))

import dataset
import dataset_loaders
import neural_networks.feed_forward_neural_network as ffnn

FLAGS = tf.app.flags.FLAGS

# training constants
tf.app.flags.DEFINE_integer('batch_size', 
                            32,
                            """Number of samples in a batch.""")
tf.app.flags.DEFINE_integer('num_epochs', 
                            100,
                            """Number of training epochs.""")
tf.app.flags.DEFINE_string('snapshot_dir', 
                           '../../data/snapshots/test/',
                           """Path to directory where to save weights.""")
tf.app.flags.DEFINE_string('summary_dir', 
                           '../../data/summaries/test',
                           """Path to directory where to save summaries.""")
tf.app.flags.DEFINE_string('julia_weights_filepath', 
                           '../../data/networks/test.weights',
                           """Path to file where to save julia weights.""")
tf.app.flags.DEFINE_integer('save_every', 
                            1000000,
                            """Number of epochs between network saves.""")
tf.app.flags.DEFINE_bool('verbose', 
                            True,
                            """Wether or not to print out progress.""")
tf.app.flags.DEFINE_integer('debug_size', 
                            None,
                            """Debug size to use.""")
tf.app.flags.DEFINE_integer('random_seed', 
                            1,
                            """Random seed value to use.""")
tf.app.flags.DEFINE_bool('load_network', 
                            False,
                            """Wether or not to load from a saved network.""")
tf.app.flags.DEFINE_integer('log_summaries_every', 
                            2,
                            """Number of batches between logging summaries.""")
tf.app.flags.DEFINE_integer('save_weights_every', 
                            1,
                            """Number of batches between logging summaries.""")
tf.app.flags.DEFINE_bool('balanced_class_loss', 
                            False,
                            """Whether or not to balance the classes in 
                            classification loss by reweighting.""")
tf.app.flags.DEFINE_integer('target_index', 
                            None,
                            """Target index to fit exclusively if set (zero-based).
                            This must be accompanied by setting output_dim to 1.""")

# network constants
tf.app.flags.DEFINE_integer('max_norm', 
                            100000,
                            """Maximum gradient norm.""")
tf.app.flags.DEFINE_integer('hidden_dim', 
                            64,
                            """Hidden units in each hidden layer.""")
tf.app.flags.DEFINE_integer('num_hidden_layers', 
                            2,
                            """Number of hidden layers.""")
tf.app.flags.DEFINE_integer('encoding_dim', 
                            8,
                            """Hidden units in each hidden layer.""")
tf.app.flags.DEFINE_string('hidden_layer_dims', 
                            '',
                            """Hidden layer sizes, empty list means use hidden_dim.""")
tf.app.flags.DEFINE_float('learning_rate', 
                            0.005,
                            """Initial learning rate to use.""")
tf.app.flags.DEFINE_float('decrease_lr_threshold', 
                            .001,
                            """Percent decrease in validation loss below 
                            which the learning rate will be decayed.""")
tf.app.flags.DEFINE_float('decay_lr_ratio', 
                            .95,
                            """Learning rate decay factor.""")
tf.app.flags.DEFINE_float('min_lr', 
                            .000005,
                            """Minimum learning rate value.""")
tf.app.flags.DEFINE_string('loss_type', 
                           'ce',
                           """Type of loss to use {mse, ce}.""")
tf.app.flags.DEFINE_string('task_type', 
                           'regression',
                           """Type of task {regression, classification}.""")
tf.app.flags.DEFINE_integer('num_target_bins', 
                            None,
                            """Number of bins into which to discretize targets.""")
tf.app.flags.DEFINE_float('dropout_keep_prob', 
                            1.,
                            """Probability to keep a unit in dropout.""")
tf.app.flags.DEFINE_boolean('use_batch_norm', 
                            False,
                            """Whether to use batch norm (True removes dropout).""")
tf.app.flags.DEFINE_float('l2_reg', 
                            0.0,
                            """Probability to keep a unit in dropout.""")
tf.app.flags.DEFINE_float('eps', 
                            1e-8,
                            """Minimum probability value.""")
tf.app.flags.DEFINE_float('reconstruction_weight', 
                            1.,
                            """Weight of the reconstruction loss.""")

# dataset constants
tf.app.flags.DEFINE_string('dataset_filepath',
                            '../../data/datasets/risk.jld',
                            'Filepath of dataset.')
tf.app.flags.DEFINE_integer('input_dim', 
                            166,
                            """Dimension of input.""")
tf.app.flags.DEFINE_integer('timesteps', 
                            1,
                            """Number of input timesteps.""")
tf.app.flags.DEFINE_integer('output_dim', 
                            5,
                            """Dimension of output.""")
tf.app.flags.DEFINE_bool('use_priority', 
                            False,
                            """Wether or not to use a prioritized dataset.""")
tf.app.flags.DEFINE_float('priority_alpha', 
                            0.25,
                            """Alpha parameter for prioritization.""")
tf.app.flags.DEFINE_float('priority_beta', 
                            1.0,
                            """Beta parameter for prioritization.""")

def custom_parse_flags(flags):
    if flags.hidden_layer_dims != '':
        dims = flags.hidden_layer_dims.split(' ')
        dims = [int(dim) for dim in dims]
    else:
        dims = [flags.hidden_dim for _ in range(flags.num_hidden_layers)]

    flags.hidden_layer_dims = dims
    print('Building network with hidden dimensions: {}'.format(
            flags.hidden_layer_dims))

def classification_score(y, y_pred, name, y_null=None):
    print('\nclassification results for {}'.format(name))
    for tidx in range(y.shape[1]):
        print('target: {}'.format(tidx))
        print(classification_report(y[:,tidx], y_pred[:,tidx]))
        input()

def main(argv=None):
    # custom parse of flags for list input
    custom_parse_flags(FLAGS)

    # set random seeds
    np.random.seed(FLAGS.random_seed)
    tf.set_random_seed(FLAGS.random_seed)

    # load dataset
    input_filepath = FLAGS.dataset_filepath
    data = dataset_loaders.risk_dataset_loader(
        input_filepath, shuffle=True, train_split=.9, 
        debug_size=FLAGS.debug_size, timesteps=FLAGS.timesteps,
        num_target_bins=FLAGS.num_target_bins, balanced_class_loss=FLAGS.balanced_class_loss, target_index=FLAGS.target_index)

    d = dataset.Dataset(data, FLAGS)

    print(np.mean(d.data['y_train'], axis=0))
    print(np.mean(d.data['y_val'], axis=0))
    y = copy.deepcopy(d.data['y_val'])
    y[y==0.] = 1e-8
    y[y==1.] = 1 - 1e-8
    baseline = np.mean(y, axis=0)
    ce = -np.sum(y * np.log(baseline)) + -np.sum((1 - y) * np.log(1 - baseline))
    mse = np.sum((y - baseline) ** 2)
    r2 = 1 - ((y - baseline) ** 2).sum() / ((y - y.mean(axis=0)) ** 2).sum()
    num_samples = len(y)
    print("cross entropy from outputting validation mean: {}".format(ce / num_samples))
    print("mse from outputting validation mean: {}".format(mse / num_samples))
    print("r2 from outputting validation mean: {}".format(r2))
    
    ce = -np.sum(y * np.log(y)) + -np.sum((1 - y) * np.log(1 - y))
    print("cross entropy from outputting correct values: {}".format(ce / num_samples))
    try:
        ce = -np.sum(y[:,3] * np.log(y[:,3])) + -np.sum((1 - y[:,3]) * np.log(1 - y[:,3]))
        print("hard brake cross entropy from outputting correct values: {}".format(ce / num_samples))
    except:
        pass
    # fit the model
    with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as session:

        network = ffnn.RiskFeatureNeuralNetwork(session, FLAGS)
        network.fit(d)

        y_idxs = np.where(np.sum(data['y_val'][:10000], axis=1) > 1e-4)[0]
        y_idxs = np.random.permutation(y_idxs)[:10]
        y_pred = network.predict(data['x_val'][y_idxs])

        # final train loss
        y_pred = network.predict(data['x_train'])
        y = data['y_train']
        y_null = np.mean(y, axis=0)
        classification_score(y, y_pred, 'train', y_null=y_null)

        # final validation loss
        y_pred = network.predict(data['x_val'])
        y = data['y_val']
        y_null = np.mean(y, axis=0)
        classification_score(y, y_pred, 'val', y_null=y_null)

if __name__ == '__main__':
    tf.app.run()
