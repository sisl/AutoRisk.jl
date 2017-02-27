
import collections
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(suppress=True, precision=8)
import os
import seaborn as sns
import sys
import tensorflow as tf
import time

path = os.path.join(os.path.dirname(__file__), os.pardir)
sys.path.append(os.path.abspath(path))
path = os.path.join(os.path.dirname(__file__), os.pardir, 'neural_networks')
sys.path.append(os.path.abspath(path))

import dataset_loaders
import initializers

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('num_epochs', 50, '')
tf.app.flags.DEFINE_integer('batch_size', 32, '')
tf.app.flags.DEFINE_integer('input_dim', 204, '')
tf.app.flags.DEFINE_float('learning_rate', 0.001, '')
tf.app.flags.DEFINE_float('dropout_keep_prob', 1., '')
tf.app.flags.DEFINE_string('encode_dims', '64 32', '')
tf.app.flags.DEFINE_string('decode_dims', '64', '')
tf.app.flags.DEFINE_string('dataset_filepath', 
    '/Users/wulfebw/Dropbox/File_Transfers/risk/risk_10_sec_10_timesteps.h5', '')
tf.app.flags.DEFINE_boolean('load_network', False, '')
tf.app.flags.DEFINE_string('snapshot_dir', 
    '../../data/snapshots/prediction_features', '')
tf.app.flags.DEFINE_integer('debug_size', None, '')
tf.app.flags.DEFINE_integer('save_weights_every', 100, '')

def custom_parse_flags(flags):
    flags.encode_dims = [int(d) for d in flags.encode_dims.split(' ')]
    flags.decode_dims = [int(d) for d in flags.decode_dims.split(' ')]

def AE(input_dim, encode_dims, decode_dims, learning_rate):

    input_ph = tf.placeholder(tf.float32,
                shape=(None, input_dim),
                name="input_ph")
    dropout_ph = tf.placeholder(tf.float32,
                shape=(),
                name="dropout_ph")

    weights_initializer = initializers.weights_initializer = initializers.get_weight_initializer(
            'relu')
    bias_initializer = initializers.bias_initializer = initializers.get_bias_initializer(
            'relu')

    encode = input_ph
    for (lidx, hidden_dim) in enumerate(encode_dims):
        if lidx == len(encode_dims) - 1:
            encode = tf.contrib.layers.fully_connected(encode, hidden_dim, activation_fn=None)
        else:
            encode = tf.contrib.layers.fully_connected(
                encode, 
                hidden_dim, 
                weights_initializer=weights_initializer,
                biases_initializer=bias_initializer,
                activation_fn=tf.nn.relu)
            encode = tf.nn.dropout(encode, dropout_ph)

    decode = encode
    decode_dims += [input_dim]
    for (lidx, hidden_dim) in enumerate(decode_dims):
        if lidx == len(encode_dims) - 1:
            decode = tf.contrib.layers.fully_connected(decode, hidden_dim, activation_fn=None)
        else:
            decode = tf.contrib.layers.fully_connected(
                decode, 
                hidden_dim, 
                weights_initializer=weights_initializer,
                biases_initializer=bias_initializer,
                activation_fn=tf.nn.relu)
            decode = tf.nn.dropout(decode, dropout_ph)

    loss = tf.reduce_sum((input_ph - decode) ** 2)
    opt = tf.train.AdamOptimizer(learning_rate)   
    train_op = opt.minimize(loss)

    return input_ph, dropout_ph, encode, decode, loss, train_op

def plot_features(data, session, input_ph,  dropout_ph, encode, flags):
    num_train_batches = int(len(data['x_train']) / flags.batch_size)
    if (num_train_batches * flags.batch_size) < len(data['x_train']):
        num_train_batches += 1
    num_samples = len(data['x_train'])
    encodings = np.empty((num_samples, flags.encode_dims[-1]))
    for bidx in range(num_train_batches):
        s = bidx * flags.batch_size
        e = s + flags.batch_size
        x = data['x_train'][s:e]
        feed_dict = {input_ph: x, dropout_ph: 1.}
        num_encode = session.run(encode, feed_dict=feed_dict)
        encodings[s:e,:] = num_encode

    counts = collections.defaultdict(int)
    colors, cat_max, idxs = [], 500, []
    targets = data['y_train']
    for i, t in enumerate(targets):
        c = None
        if t[0] > 0. and counts[0] < cat_max:
            c = 'blue'
            counts[0] += 1
        elif t[1] > 0. and counts[1] < cat_max:
            c = 'red'
            counts[1] += 1
        elif t[2] > 0. and counts[2] < cat_max:
            c = 'purple'
            counts[2] += 1
        elif t[3] > 0. and counts[3] < cat_max:
            c = 'orange'
            counts[3] += 1
        elif t[4] > 0. and counts[4] < cat_max:
            c = 'green'
            counts[4] += 1
        elif counts[5] < cat_max:
            c = 'black'
            counts[5] += 1
        if c is not None:
            colors.append(c)
            idxs.append(i)
    plt.figure(figsize=(10,10))
    plt.scatter(encodings[idxs,0], encodings[idxs,1], c=colors, alpha=.5)
    plt.show()

def main(argv=None):
    # parse the hidden layer dims
    custom_parse_flags(FLAGS)

    # load the data
    data = dataset_loaders.risk_dataset_loader(
        FLAGS.dataset_filepath, shuffle=False, train_split=.9, 
        debug_size=FLAGS.debug_size, timesteps=1)

    # only select the indices with nonzero targets
    idxs = np.where(np.sum(data['y_train'], axis=1) > 0)[0]
    idxs = np.array(list(idxs) + list(range(500)))
    data['x_train'] = data['x_train'][idxs]
    data['y_train'] = data['y_train'][idxs]
    idxs = np.where(np.sum(data['y_val'], axis=1) > 0)[0]
    idxs = np.array(list(idxs) + list(range(500)))
    data['x_val'] = data['x_val'][idxs]
    data['y_val'] = data['y_val'][idxs]

    print(len(data['x_train']))

    with tf.Session() as session:
        # build the autoencoder
        input_ph, dropout_ph, encode, decode, loss, train_op = AE(
            FLAGS.input_dim, FLAGS.encode_dims, FLAGS.decode_dims, 
            FLAGS.learning_rate)
        session.run(tf.global_variables_initializer())


        saver = tf.train.Saver(max_to_keep=10)
        if not os.path.exists(FLAGS.snapshot_dir):
            os.mkdir(FLAGS.snapshot_dir)
        if FLAGS.load_network:
            filepath = tf.train.latest_checkpoint(FLAGS.snapshot_dir)
            if filepath is not None:
                saver.restore(session, filepath)

        # train it
        num_train_batches = int(len(data['x_train']) / FLAGS.batch_size)
        if (num_train_batches * FLAGS.batch_size) < len(data['x_train']):
            num_train_batches += 1
        num_val_batches = int(len(data['x_val']) / FLAGS.batch_size)
        if num_val_batches * FLAGS.batch_size < len(data['x_val']):
            num_val_batches += 1
        start_time = time.time()
        for epoch in range(FLAGS.num_epochs):

            # train
            train_losses = []
            for bidx in range(num_train_batches):
                s = bidx * FLAGS.batch_size
                e = s + FLAGS.batch_size
                x = data['x_train'][s:e]
                feed_dict = {
                    input_ph: x,
                    dropout_ph: FLAGS.dropout_keep_prob
                }
                output_list = [loss, train_op]
                num_loss, _ = session.run(output_list, feed_dict=feed_dict)
                train_losses.append(num_loss / len(x))

            # val
            val_losses = []
            for bidx in range(num_val_batches):
                s = bidx * FLAGS.batch_size
                e = s + FLAGS.batch_size
                x = data['x_val'][s:e]
                feed_dict = {input_ph: x, dropout_ph: 1.}
                num_loss = session.run(loss, feed_dict=feed_dict)
                val_losses.append(num_loss / len(x))

            if (epoch + 1) % FLAGS.save_weights_every == 0:
                filepath = os.path.join(FLAGS.snapshot_dir, 'weights')
                saver.save(session, filepath, global_step=epoch)

            # report
            print('epoch: {}\ttrain: {}\tval: {}\ttime: {}'.format(
                epoch, np.mean(train_losses), np.mean(val_losses), 
                time.time() - start_time))

        plot_features(data, session, input_ph,  dropout_ph, encode, FLAGS)

if __name__ == '__main__':
    tf.app.run()