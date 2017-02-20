
import copy 
import numpy as np
import os
import sys
import tensorflow as tf
import unittest

path = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'scripts')
sys.path.append(os.path.abspath(path))
path = os.path.join(os.path.dirname(__file__), os.pardir)
sys.path.append(os.path.abspath(path))

import dataset
import neural_networks.recurrent_neural_network as rnn
import testing_flags
import testing_utils

class TestRecurrentNeuralNetwork(unittest.TestCase):

    def setUp(self):
        # reset graph before each test case
        tf.python.reset_default_graph()

    def test_fit_basic(self):
        # goal is to overfit a simple dataset
        tf.set_random_seed(1)
        np.random.seed(1)
        flags = testing_flags.FLAGS
        flags.input_dim = 1
        flags.timesteps = 3
        flags.hidden_dim = 8
        flags.num_hidden_layers = 1
        flags.hidden_layer_dims = [8]
        flags.output_dim = 1
        flags.batch_size = 50
        flags.num_epochs = 50
        flags.learning_rate = .005
        flags.l2_reg = 0.0
        flags.dropout_keep_prob = 1.
        flags.verbose = False
        flags.save_weights_every = 100000
        flags.snapshot_dir = os.path.abspath(os.path.join(
            os.path.dirname(__file__), os.pardir, 'data','snapshots','test'))

        # each sample is a sequence of random normal values
        # if sum of inputs < 0 -> output is 0
        # if sum of inputs > 0 -> output is 1
        num_samples = 500
        x = np.random.randn(num_samples * flags.timesteps * flags.input_dim)
        x = x.reshape(-1, flags.timesteps, flags.input_dim)
        z = np.sum(x, axis=(1,2))
        hi_idxs = np.where(z > .5)[0]
        lo_idxs = np.where(z < -.5)[0]
        idxs = np.array(list(hi_idxs) + list(lo_idxs))
        z = z[idxs]
        x = x[idxs]
        y = np.ones((len(z), 1))
        y[z < 0] = 0

        data = {
            'x_train': x,
            'y_train': y,
            'x_val': x,
            'y_val': y
        }
        d = dataset.Dataset(data, flags)
        with tf.Session() as session:
            network = rnn.RecurrentNeuralNetwork(session, flags)
            network.fit(d)
            actual = network.predict(x)
            actual[actual < .5] = 0
            actual[actual >= .5] = 1
            np.testing.assert_array_almost_equal(y, actual, 8)

if __name__ == '__main__':
    unittest.main()