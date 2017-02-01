
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
import neural_networks.feed_forward_neural_network as ffnn
import testing_flags
import testing_utils

class TestFeedForwardNeuralNetwork(unittest.TestCase):

    def setUp(self):
        # reset graph before each test case
        tf.python.reset_default_graph()

    def test_init(self):
        flags = testing_flags.FLAGS
        with tf.Session() as session:
            ffnn.FeedForwardNeuralNetwork(session, flags)

    def test_build_network(self):
        flags = testing_flags.FLAGS
        flags.input_dim = 3
        flags.hidden_dim = 6
        flags.num_hidden_layers = 2
        flags.output_dim = 2
        flags.batch_size = 6
        flags.save_weights_every = 100000
        with tf.Session() as session:
            network = ffnn.FeedForwardNeuralNetwork(session, flags)

            # check scores for zero weights
            testing_utils.assign_trainable_variables_to_constant(constant=0)
            inputs = np.zeros((flags.batch_size, flags.input_dim))
            actual_outputs = session.run(
                network._scores, 
                feed_dict={network._input_ph: inputs, network._dropout_ph: 1., network._lr_ph: flags.learning_rate})
            expected_outputs = np.zeros((flags.batch_size, flags.output_dim))
            np.testing.assert_array_equal(expected_outputs, actual_outputs)

            # check for scores for ones weights, zero input
            testing_utils.assign_trainable_variables_to_constant(constant=1)
            actual_outputs = session.run(
                network._scores, 
                feed_dict={network._input_ph: inputs, network._dropout_ph: 1., network._lr_ph: flags.learning_rate})
            output_value = 43
            expected_outputs = np.ones((flags.batch_size, flags.output_dim)) * output_value
            np.testing.assert_array_equal(expected_outputs, actual_outputs)

            # check for scores for ones weights, negative one input
            testing_utils.assign_trainable_variables_to_constant(constant=1)
            inputs = np.ones((flags.batch_size, flags.input_dim)) * -1
            actual_outputs = session.run(
                network._scores, 
                feed_dict={network._input_ph: inputs, network._dropout_ph: 1., network._lr_ph: flags.learning_rate})
            output_value = 7
            expected_outputs = np.ones((flags.batch_size, flags.output_dim)) * output_value
            np.testing.assert_array_equal(expected_outputs, actual_outputs)

    def test_ability_to_overfit_debug_dataset(self):
        """
        Description:
            - Test network's ability to overfit a small dataset independent
                of whether 'fit' is correctly implemented.
        """
        tf.set_random_seed(1)
        np.random.seed(1)
        flags = testing_flags.FLAGS
        flags.input_dim = 3
        flags.hidden_dim = 16
        flags.num_hidden_layers = 2
        flags.output_dim = 2
        flags.batch_size = 12
        flags.num_epochs = 200
        flags.learning_rate = .05
        flags.save_weights_every = 100000
        with tf.Session() as session:
            network = ffnn.FeedForwardNeuralNetwork(session, flags)
            # data set maps 1->0 and 0->1
            x = np.vstack(
                (np.ones((flags.batch_size // 2, flags.input_dim)), 
                  np.zeros((flags.batch_size // 2, flags.input_dim))))
            y = np.vstack(
                (np.zeros((flags.batch_size // 2, flags.output_dim)), 
                    np.ones((flags.batch_size // 2, flags.output_dim))))

            # training epochs
            for epoch in range(flags.num_epochs):
                loss, probs, scores, _ = session.run([network._loss, network._probs, network._scores, network._train_op],
                    feed_dict={network._input_ph: x, network._target_ph: y,
                    network._dropout_ph: network.flags.dropout_keep_prob, network._lr_ph: flags.learning_rate})

            probs = session.run(network._probs,
                    feed_dict={network._input_ph: x, network._target_ph: y,
                    network._dropout_ph: 1., network._lr_ph: flags.learning_rate})

            self.assertAlmostEqual(loss, 0, 4)
            np.testing.assert_array_almost_equal(probs, y, 4)

    def test_predict(self):
        flags = testing_flags.FLAGS
        flags.input_dim = 3
        flags.hidden_dim = 6
        flags.num_hidden_layers = 2
        flags.output_dim = 2
        flags.batch_size = 6
        flags.save_weights_every = 100000

        x = np.vstack(
            (np.zeros((flags.batch_size // 2, flags.input_dim)), 
                -1 * np.ones((flags.batch_size // 2, flags.input_dim))))
        with tf.Session() as session:
            network = ffnn.FeedForwardNeuralNetwork(session, flags)
            testing_utils.assign_trainable_variables_to_constant(constant=1)
            actual = network.predict(x)
            expected = np.ones((flags.batch_size, flags.output_dim))
            expected[:flags.batch_size // 2] *= testing_utils.sigmoid(43)
            expected[flags.batch_size // 2:] *= testing_utils.sigmoid(7)
            np.testing.assert_array_almost_equal(expected, actual, 4)
            
    def test_fit_basic(self):
        """
        Description:
            - overfit a debug dataset using the 'fit' method
        """
        tf.set_random_seed(1)
        np.random.seed(1)
        flags = testing_flags.FLAGS
        flags.input_dim = 3
        flags.hidden_dim = 32
        flags.num_hidden_layers = 2
        flags.output_dim = 2
        flags.batch_size = 16
        flags.num_epochs = 200
        flags.learning_rate = .05
        flags.l2_reg = 0.0
        flags.save_weights_every = 100000
        flags.snapshot_dir = os.path.abspath(os.path.join(
            os.path.dirname(__file__), os.pardir, 'data','snapshots','test'))

        x = np.vstack(
            (np.ones((flags.batch_size // 2, flags.input_dim)), 
                -1 * np.ones((flags.batch_size // 2, flags.input_dim))))
        y = np.vstack(
            (np.zeros((flags.batch_size // 2, flags.output_dim)), 
                np.ones((flags.batch_size // 2, flags.output_dim))))
        data = {'x_train': x,
            'y_train': y,
            'x_val': x,
            'y_val': y}
        d = dataset.Dataset(data, flags)
        with tf.Session() as session:
            network = ffnn.FeedForwardNeuralNetwork(session, flags)
            network.fit(d)
            actual = network.predict(x)
            np.testing.assert_array_almost_equal(y, actual, 8)

    def test_fit_complex(self):
        tf.set_random_seed(1)
        np.random.seed(1)
        flags = testing_flags.FLAGS
        flags.input_dim = 1
        flags.hidden_dim = 32
        flags.num_hidden_layers = 2
        flags.output_dim = 1
        flags.batch_size = 32
        flags.num_epochs = 8
        flags.learning_rate = .002
        flags.l2_reg = 0.0
        flags.verbose = False
        flags.save_weights_every = 100000
        flags.snapshot_dir = os.path.abspath(os.path.join(
            os.path.dirname(__file__), os.pardir, 'data','snapshots','test'))

        x = np.random.randn(1000).reshape(-1, 1)
        y = np.ones(x.shape)
        y[x < 0] = 0
        data = {'x_train': x,
            'y_train': y,
            'x_val': x,
            'y_val': y}
        d = dataset.Dataset(data, flags)
        with tf.Session() as session:
            network = ffnn.FeedForwardNeuralNetwork(session, flags)
            network.fit(d)
            actual = network.predict(x)
            actual[actual < .5] = 0
            actual[actual >= .5] = 1
            np.testing.assert_array_almost_equal(y, actual, 8)

if __name__ == '__main__':
    unittest.main()