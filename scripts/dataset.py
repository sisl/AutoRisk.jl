"""
Class for iterating through a dataset.
"""

import numpy as np

class Dataset(object):

    def __init__(self, data, flags):
        """
        Description:
            - Initialize the dataset.

        Args:
            - data: a dictionary that must contain the keys:
                'x_train', 'y_train', 'x_val', 'y_val'
                Each of these keys should correspond to a value,
                such that when indexing into the first dimension
                of that value, you retrieve an element of a sample.
            - flags: object containing options
        """
        self.flags = flags
        self.data = data

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        keys = ['x_train', 'y_train', 'x_val', 'y_val']
        for k in keys:
            if k not in data:
                raise ValueError('data must contain key: {}'.format(k))

        # set data
        self._data = data

        # compute batch information
        for split in ['train', 'val']:
            num_samples = len(data['x_{}'.format(split)])
            num_batches = int(num_samples / self.flags.batch_size)

            # if num_samples not divisible by batch_size, then 
            # simply add an additional batch, which will be addressed
            # in next_batch using python indexing past the end of a container
            if num_samples % self.flags.batch_size != 0:
                num_batches += 1

            if split == 'train':
                self.num_train_batches = num_batches
            else:
                self.num_val_batches = num_batches

    def next_batch(self, validation=False):
        # retrieve training or validation set
        suffix = 'val' if validation else 'train'
        x, y = self.data['x_{}'.format(suffix)], self.data['y_{}'.format(suffix)]
        num_batches = self.num_val_batches if validation else self.num_train_batches

        # suffle data for this epoch
        idxs = np.random.permutation(len(x))
        x = x[idxs]
        y = y[idxs]

        # yield data in batches
        for bidx in range(num_batches):
            # compute start and end indices
            start = bidx * self.flags.batch_size
            end = (bidx + 1) * self.flags.batch_size

            # retrieve the data
            inputs = x[start:end]
            targets = y[start:end]

            yield inputs, targets

class WeightedDataset(object):

    def __init__(self, data, flags):
        """
        Description:
            - Initialize the dataset.

        Args:
            - data: a dictionary that must contain the keys:
                'x_train', 'y_train', 'x_val', 'y_val'
                Each of these keys should correspond to a value,
                such that when indexing into the first dimension
                of that value, you retrieve an element of a sample.
            - flags: object containing options
        """
        self.flags = flags
        self.data = data

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        keys = ['x_train', 'y_train', 'w_train', 'x_val', 'y_val', 'w_val']
        for k in keys:
            if k not in data:
                raise ValueError('data must contain key: {}'.format(k))

        # set data
        self._data = data

        # compute batch information
        for split in ['train', 'val']:
            num_samples = len(data['x_{}'.format(split)])
            num_batches = int(num_samples / self.flags.batch_size)

            # if num_samples not divisible by batch_size, then 
            # simply add an additional batch, which will be addressed
            # in next_batch using python indexing past the end of a container
            if num_samples % self.flags.batch_size != 0:
                num_batches += 1

            if split == 'train':
                self.num_train_batches = num_batches
            else:
                self.num_val_batches = num_batches

    def next_batch(self, validation=False):
        # retrieve training or validation set
        suffix = 'val' if validation else 'train'
        x, y, w = (self.data['x_{}'.format(suffix)], 
            self.data['y_{}'.format(suffix)], self.data['w_{}'.format(suffix)])
        num_batches = self.num_val_batches if validation else self.num_train_batches

        # suffle data for this epoch
        idxs = np.random.permutation(len(x))
        x = x[idxs]
        y = y[idxs]
        w = w[idxs]

        # yield data in batches
        for bidx in range(num_batches):
            # compute start and end indices
            start = bidx * self.flags.batch_size
            end = (bidx + 1) * self.flags.batch_size
            yield x[start:end], y[start:end], w[start:end]
        