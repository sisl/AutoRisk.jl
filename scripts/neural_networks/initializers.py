"""
Functions for defining variables in a network.
"""

import tensorflow as tf

def get_weight_initializer(activation):
    """
    Description:
        - Given an activation function, return a weight
            initializer that works well for that activation
            function.

            Relu: "Delving Deep into Rectifiers:
            Surpassing Human-Level Performance on ImageNet
            Classification" 
            Source: https://arxiv.org/abs/1502.01852

            Tanh: 

    Args:
        - activation: string indicating the activation function.
            one of {'relu', 'tanh'}

    Returns:
        - initializer: a tensorflow weight initializer.
    """
    if activation == 'relu':
        initializer = tf.contrib.layers.variance_scaling_initializer(
            factor=2.0, mode='FAN_IN', uniform=False)
    elif activation == 'tanh':
        initializer = tf.contrib.layers.variance_scaling_initializer(
            factor=1.0, mode='FAN_AVG', uniform=False)
    else:
        raise ValueError('invalid activation: {}'.format(activation))
    return initializer

def get_bias_initializer(activation):
    """
    Description: 
        - Given an activation function, return a bias
            initializer that works well for that activation
            function.

    Args:
        - activation: string denoting activation, 
            one of {'relu', 'tanh'}

    Returns:
        - initializer: tensorflow bias initializer
    """
    if activation == 'relu':
        initializer = tf.constant_initializer(0.1)
    elif activation == 'tanh':
        initializer = tf.constant_initializer(0.0)
    else:
        raise ValueError('invalid activation: {}'.format(activation))
    return initializer
