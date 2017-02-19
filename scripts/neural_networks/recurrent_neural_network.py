"""
A recurrent neural network class
"""
import collections
import numpy as np
import os
import tensorflow as tf
import time

from . import initializers
from .feed_forward_neural_network import NeuralNetwork

class RecurrentNeuralNetwork(NeuralNetwork):
    def __init__(self, session, flags):
        """
        Description:
            - Initializes this network by storing the tf flags and 
                building the model.

        Args:
            - session: the session with which to execute model operations
            - flags: tensorflow flags object containing network options
        """
        super(RecurrentNeuralNetwork, self).__init__(session, flags)

    def _build_placeholders(self):
        """
        Description:
            - build placeholders for inputs to the tf graph.

        Returns:
            - input_ph: placeholder for a input batch
            - target_ph: placeholder for a target batch
            - dropout_ph: placeholder for fraction of activations 
                to drop
        """
        input_ph = tf.placeholder(tf.float32,
                shape=(None, self.flags.timesteps, self.flags.input_dim),
                name="input_ph")
        target_ph = tf.placeholder(tf.float32,
                shape=(None, self.flags.output_dim),
                name="target_ph")
        dropout_ph = tf.placeholder(tf.float32,
                shape=(),
                name="dropout_ph")
        learning_rate_ph = tf.placeholder(tf.float32, 
                shape=(),
                name="lr_ph")

        # summaries
        tf.summary.scalar('dropout keep prob', dropout_ph)
        tf.summary.scalar('learning_rate', learning_rate_ph)

        return input_ph, target_ph, dropout_ph, learning_rate_ph

    def _build_network(self, input_ph, dropout_ph):
        """
        Description:
            - Builds a recurrent neural network where the features are first
                mapped from the input dim to the hidden dim of the RNN by a 
                feed forward network.

        Args:
            - input_ph: placeholder for the inputs
                shape = (batch_size, input_dim)
            - dropout_ph: placeholder for dropout value

        Returns:
            - scores: the scores for the target values
        """

        # build initializers specific to relu
        weights_initializer = initializers.get_weight_initializer(
            'relu')
        bias_initializer = initializers.get_bias_initializer(
            'relu')

        # build regularizers
        weights_regularizer = tf.contrib.layers.l2_regularizer(
            self.flags.l2_reg)

        # build hidden layers for feed forward network 
        # if layer dims not set individually, then assume all the same dim
        hidden_layer_dims = self.flags.hidden_layer_dims
        if len(hidden_layer_dims) == 0:
            hidden_layer_dims = [self.flags.hidden_dim 
                for _ in range(self.flags.num_hidden_layers)]

        hidden = input_ph
        for (lidx, hidden_dim) in enumerate(hidden_layer_dims):
            hidden = tf.contrib.layers.fully_connected(hidden, 
                hidden_dim, 
                activation_fn=tf.nn.relu,
                weights_initializer=weights_initializer,
                weights_regularizer=weights_regularizer,
                biases_initializer=bias_initializer)
            # tf.histogram_summary("layer_{}_activation".format(lidx), hidden)
            hidden = tf.nn.dropout(hidden, dropout_ph)

        # build recurrent network 
        cell = tf.nn.rnn_cell.GRUCell(num_units=hidden_layer_dims[-1])
        outputs, states = tf.nn.dynamic_rnn(
            cell=cell,
            dtype=tf.float32,
            inputs=hidden)

        # build output layer
        last_output = tf.squeeze(tf.slice(
            outputs, (0, self.flags.timesteps - 1, 0), (-1,-1,-1)), 1)
        scores = tf.contrib.layers.fully_connected(last_output, 
                self.flags.output_dim, 
                activation_fn=None,
                weights_regularizer=weights_regularizer)

        return scores