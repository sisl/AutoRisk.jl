import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
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

class TestFeedForwardNeuralNetworkMNIST(unittest.TestCase):

    def setUp(self):
        # reset graph before each test case
        tf.python.reset_default_graph()

    def test_fit_mnist(self):
        # set flags
        tf.set_random_seed(1)
        np.random.seed(1)
        flags = testing_flags.FLAGS
        flags.input_dim = 28 * 28
        flags.hidden_dim = 256
        flags.num_hidden_layers = 3
        flags.output_dim = 10
        flags.batch_size = 8
        flags.num_epochs = 20
        flags.learning_rate = .001
        flags.l2_reg = 0.0
        flags.dropout_keep_prob = .5
        flags.verbose = False
        flags.save_weights_every = 100000

        # load data
        data = testing_utils.load_mnist(debug_size=1000)
        d = dataset.Dataset(data, flags)

        # build network
        with tf.Session() as session:
            network = ffnn.FeedForwardNeuralNetwork(session, flags)
            network.fit(d)

            y_pred = network.predict(data['x_val'])
            y_pred = np.argmax(y_pred, axis=1)
            y = np.argmax(data['y_val'], axis=1)
            acc = len(np.where(y_pred == y)[0]) / float(len(y_pred))

            # check that validation accuracy is above 90%
            self.assertTrue(acc > .9)

            # if run solo, then display some images and predictions
            if __name__ == '__main__':
                for num_samp in range(1):
                    x = data['x_train'][num_samp].reshape(1, -1)
                    print(np.argmax(network.predict(x)[0], axis=0))
                    print(np.argmax(data['y_train'][num_samp], axis=0))
                    plt.imshow(data['x_train'][num_samp].reshape(28,28))
                    plt.show()
                    print('\n')
                    x = data['x_val'][num_samp].reshape(1, -1)
                    print(np.argmax(network.predict(x)[0], axis=0))
                    print(np.argmax(data['y_val'][num_samp], axis=0))
                    plt.imshow(data['x_val'][num_samp].reshape(28,28))
                    plt.show()



if __name__ == '__main__':
    unittest.main()