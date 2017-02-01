import numpy as np
np.set_printoptions(suppress=True, precision=4)
import os
import sys
import tensorflow as tf

path = os.path.join(os.path.dirname(__file__), os.pardir)
sys.path.append(os.path.abspath(path))

import compression.run_compression
import dataset
import dataset_loaders
import neural_networks.feed_forward_neural_network as ffnn
import neural_networks.utils

FLAGS = compression.run_compression.FLAGS

def main(argv=None):
    
    # set random seeds
    np.random.seed(FLAGS.random_seed)
    tf.set_random_seed(FLAGS.random_seed)

    # load dataset
    input_filepath = FLAGS.dataset_filepath
    data = dataset_loaders.risk_dataset_loader(
        input_filepath, shuffle=True, train_split=.8, 
        debug_size=FLAGS.debug_size)
    
    # set training sizes
    num_runs = 12
    train_scores = []
    val_scores = []
    sizes = [int(v) for v in np.logspace(np.log2(1000), np.log2(len(data['x_train'])), num_runs, base=2.0)]
    print(sizes)
    eps = 1e-12
    
    # for each size, fit for a set number of epochs and then compute loss
    for i in range(num_runs):
        # train for longer with more data
        FLAGS.num_epochs += 20

        # create run-specific dataset
        cur_data = {'x_train': data['x_train'][:sizes[i]],
                    'y_train': data['y_train'][:sizes[i]],
                    'x_val': data['x_val'],
                    'y_val': data['y_val']}
        d = dataset.Dataset(cur_data, FLAGS)

        with tf.Session() as session:
            network = ffnn.FeedForwardNeuralNetwork(session, FLAGS)
            network.fit(d)
        
            # final train loss
            y_pred = network.predict(cur_data['x_train']).astype(np.float128)
            y_pred[y_pred < eps] = eps
            y_pred[y_pred > (1 - eps)] = 1 - eps
            y = cur_data['y_train']
            ce = (-np.sum(y * np.log(y_pred)) + -np.sum((1 - y) * np.log(1 - y_pred))) / len(y)
            mse = np.mean((y - y_pred) ** 2)
            train_scores.append((ce, mse))

            # final validation loss
            y_pred = network.predict(cur_data['x_val']).astype(np.float128)
            y_pred[y_pred < eps] = eps
            y_pred[y_pred > (1 - eps)] = 1 - eps
            y = cur_data['y_val']
            ce = (-np.sum(y * np.log(y_pred)) + -np.sum((1 - y) * np.log(1 - y_pred))) / len(y)
            mse = np.mean((y - y_pred) ** 2)
            np.savez('../../media/scratch.npz',y_pred=y_pred)
            val_scores.append((ce, mse))

            print('size: {}\ttrain: {}\tval: {}'.format(sizes[i], train_scores[i], val_scores[i]))

        # reset graph after each run
        tf.python.reset_default_graph()
        output_filepath = os.path.join(
            '../../media/learning_curves/', os.path.split(input_filepath)[-1].replace('.h5','.npz'))
        np.savez(output_filepath, sizes=sizes, train_scores=train_scores, val_scores=val_scores)
        
if __name__ == '__main__':
    tf.app.run()


