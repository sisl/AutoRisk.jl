
import numpy as np
import os
import sys
import unittest

path = os.path.join(os.path.dirname(__file__), os.pardir, 'scripts')
sys.path.append(os.path.abspath(path))

import testing_flags
import dataset

def get_debug_data(flags, train_samples=10, val_samples=5):
    data = {'x_train': np.zeros((train_samples, flags.input_dim)),
        'y_train': np.zeros((train_samples, flags.output_dim)),
        'x_val': np.ones((val_samples, flags.input_dim)),
        'y_val': np.ones((val_samples, flags.output_dim))}
    return data

class TestDataset(unittest.TestCase):

    def test_init(self):
        flags = testing_flags.FLAGS
        data = get_debug_data(flags)
        d = dataset.Dataset(data, flags)

    def test_next_batch(self):
        flags = testing_flags.FLAGS
        flags.input_dim = 3
        flags.output_dim = 2
        flags.batch_size = 5
        data = get_debug_data(flags, train_samples=11, val_samples=4)
        d = dataset.Dataset(data, flags)

        # train
        actual_data_in_batches = list(d.next_batch())
        expected_data_in_batches = [(np.zeros((5, 3)), np.zeros((5, 2))),
            (np.zeros((5, 3)), np.zeros((5, 2))), 
            (np.zeros((1, 3)), np.zeros((1, 2)))]
        for (a, e) in zip(actual_data_in_batches, expected_data_in_batches):
            np.testing.assert_array_equal(e[0], a[0])
            np.testing.assert_array_equal(e[1], a[1])

        # validation
        actual_data_in_batches = list(d.next_batch(validation=True))
        expected_data_in_batches = [(np.ones((4, 3)), np.ones((4, 2)))]
        for (a, e) in zip(actual_data_in_batches, expected_data_in_batches):
            np.testing.assert_array_equal(e[0], a[0])
            np.testing.assert_array_equal(e[1], a[1])

if __name__ == '__main__':
    unittest.main()