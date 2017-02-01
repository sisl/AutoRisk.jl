
import h5py
import numpy as np
import tensorflow as tf

def save_trainable_variables(output_filepath, session, data=None):
    """
    Description:
        - Save trainable variables to an hdf5 file.

    Args:
        - output_filepath: string filepath where to save weights
        - session: tensorflow session to use in evaluating the weights
        - data: the data on which the network was trained, used for
            storing the input mean and std dev values
    """
    variables = tf.trainable_variables()
    weight_file = h5py.File(output_filepath, 'w')
    weight_group = weight_file.create_group('weights')
    
    for var in variables:
        weight_group[var.name] = session.run(var)

    # if data provided, then also store the means and std devs
    if data is not None:
        stats_group = weight_file.create_group('stats')
        stats_group['means'] = data['means']
        stats_group['stds'] = data['stds']
        
    weight_file.close()