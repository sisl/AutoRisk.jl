"""
A feed-forward neural network class
"""
import collections
import numpy as np
import os
import tensorflow as tf
import time

from . import initializers

class NeuralNetwork(object):

    def __init__(self, session, flags):
        """
        Description:
            - Initializes this network by storing the tf flags and 
                building the model.

        Args:
            - session: the session with which to execute model operations
            - flags: tensorflow flags object containing network options
        """
        self.session = session
        self.flags = flags
        self._build_model()

        # saving and logging setup
        self.saver = tf.train.Saver(
            max_to_keep=100, keep_checkpoint_every_n_hours=.5)
        self.train_writer = tf.summary.FileWriter(
            os.path.join(self.flags.summary_dir, 'train'), 
            self.session.graph)
        self.test_writer = tf.summary.FileWriter(
            os.path.join(self.flags.summary_dir, 'val'), 
            self.session.graph)
        self.info = collections.defaultdict(list)

    def fit(self, dataset):
        """
        Description:
            - Fit this model to the provided dataset.

        Args:
            - dataset: the dataset to fit. Must implement
                the next_batch function  
        """
        self.start_time = time.time()

        # optionally load
        if self.flags.load_network:
            self.load()

        # fit the model to the dataset over a number of epochs
        for epoch in range(self.flags.num_epochs):
            train_loss, val_loss = 0, 0

            # train epoch
            for bidx, (x, y) in enumerate(dataset.next_batch()):
                summary, loss, _ = self.session.run(
                    [self._summary_op, self._loss, self._train_op],
                    feed_dict={self._input_ph: x, self._target_ph: y,
                    self._dropout_ph: self.flags.dropout_keep_prob,
                    self._lr_ph: self.flags.learning_rate})

                train_loss += loss
                if bidx % self.flags.log_summaries_every == 0:
                    self.train_writer.add_summary(summary, epoch)
            
            # validation epoch
            for bidx, (x, y) in enumerate(dataset.next_batch(validation=True)):
                summary, loss = self.session.run([self._summary_op, self._loss],
                    feed_dict={self._input_ph: x, self._target_ph: y,
                    self._dropout_ph: 1., 
                    self._lr_ph: 0.})

                val_loss += loss
                if bidx % self.flags.log_summaries_every == 0:
                    self.test_writer.add_summary(summary, epoch)


            # print out progress if verbose
            if self.flags.verbose:
                self.log(epoch, dataset, train_loss, val_loss)

            # snapshot network
            self.save(epoch)

            # update hyperparameters
            self.update()

    def predict(self, inputs):
        """
        Description:
            - Predict output values for a set of inputs.

        Args:
            - inputs: input values to predict
                shape = (?, input_dim)

        Returns:
            - returns probability values for each output.
        """
        num_samples = len(inputs)
        outputs = np.empty((num_samples, self.flags.output_dim))
        num_batches = int(num_samples / self.flags.batch_size)
        if num_batches * self.flags.batch_size < num_samples:
            num_batches += 1
        for bidx in range(num_batches):
            s = bidx * self.flags.batch_size
            e = s + self.flags.batch_size
            batch = inputs[s:e]
            outputs[s:e, :] = self.session.run(
                self._probs, feed_dict={self._input_ph: inputs[s:e],
                self._dropout_ph: 1.})
        return outputs

    def save(self, epoch):
        """
        Description:
            - Save the session and network parameters to checkpoint file.

        Args:
            - epoch: epoch of save
        """
        if epoch % self.flags.save_weights_every == 0:
            if not os.path.exists(self.flags.snapshot_dir):
                os.mkdir(self.flags.snapshot_dir)
            filepath = os.path.join(self.flags.snapshot_dir, 'weights')
            self.saver.save(self.session, filepath, global_step=epoch)

    def load(self):
        """
        Description:
            - Load the lastest checkpoint file if it exists.
        """
        filepath = tf.train.latest_checkpoint(self.flags.snapshot_dir)
        if filepath is not None:
            self.saver.restore(self.session, filepath)

    def log(self, epoch, dataset, train_loss, val_loss):
        """
        Description:
            - Log training information to console

        Args:
            - epoch: training epoch
            - dataset: dataset used for training
            - train_loss: total training loss of the epoch
            - val_loss: total validation loss of the epoch
        """
        self.info['val_loss'].append(val_loss)
        train_loss /= len(dataset.data['x_train'])
        val_loss /= len(dataset.data['x_val']) 
        print('epoch: {}\ttrain loss: {:.6f}\tval loss: {:.6f}\ttime: {:.4f}'.format(
            epoch, train_loss, val_loss, time.time() - self.start_time))

    def update(self):

        # require at least 5 validation losses before computing the decrease
        if len(self.info['val_loss']) > 5:
            # if precentage decrease in validation loss is below a threshold
            # then reduce the learning rate
            past_loss = np.mean(self.info['val_loss'][-5:-1])
            cur_loss = self.info['val_loss'][-1]
            decrease = (past_loss - cur_loss) / past_loss
            if decrease < self.flags.decrease_lr_threshold:
                self.flags.learning_rate *= self.flags.decay_lr_ratio
                self.flags.learning_rate = max(
                    self.flags.learning_rate, self.flags.min_lr)

    def _build_model(self):
        """
        Description:
            - Builds the model, which entails defining placeholders, 
                a network, loss function, and train op. 

                Class variables created during this call all are assigned 
                to self in the body of this function, so everything that is 
                stored should be apparent from looking at this function.

                The results of these methods are passed in / out explicitly.
        """
        # placeholders
        (self._input_ph, self._target_ph, 
            self._dropout_ph, self._lr_ph) = self._build_placeholders()

        # network
        self._scores = self._build_network(
            self._input_ph, self._dropout_ph)

        # loss
        self._loss, self._probs = self._build_loss(
            self._scores, self._target_ph)

        # train operation
        self._train_op = self._build_train_op(self._loss, self._lr_ph)

        # summaries
        self._summary_op = tf.summary.merge_all()

        # intialize the model
        self.session.run(tf.global_variables_initializer())

    def _build_loss(self, scores, targets):
        """
        Description:
            - Build a loss function to optimize using the 
                scores of the network (unnormalized) and 
                the target values

        Args:
            - scores: unnormalized scores output from the network
                shape = (batch_size, output_dim)
            - targets: the target values
                shape = (batch_size, output_dim)

        Returns:
            - symbolic loss value
        """

        # create op for probability to use in 'predict'
        probs = tf.sigmoid(scores)

        # create loss separately
        if self.flags.loss_type == 'ce':
            losses = tf.reduce_sum(
                tf.nn.sigmoid_cross_entropy_with_logits(scores, targets),
                reduction_indices=(0))
        elif self.flags.loss_type == 'mse':
            losses = tf.reduce_sum((scores - targets) ** 2, 
                reduction_indices=(0))
            probs = tf.clip_by_value(scores, 0., 1.)
        elif self.flags.loss_type == 'mse_probs':
            losses = tf.reduce_sum((probs - targets) ** 2, 
                reduction_indices=(0))
        elif self.flags.loss_type == 'mse_log_probs':
            targets = tf.clip_by_value(targets, self.flags.eps, 1.)
            targets = tf.log(targets)
            losses = tf.reduce_sum((scores - targets) ** 2, 
                reduction_indices=(0))
            probs = tf.clip_by_value(tf.exp(scores), 0., 1.)
        else:
            raise(ValueError("invalid loss type: {}".format(
                self.flags.loss_type)))

        # summarize losses individually
        for tidx, target_loss in enumerate(tf.unpack(losses)):
                tf.summary.scalar('target_{}_loss'.format(tidx), target_loss) 

        # overall loss is sum of individual target losses
        loss = tf.reduce_sum(losses)

        # collect regularization losses
        reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        loss += reg_loss

        # summaries
        tf.summary.histogram('probs', probs)
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('l2 reg loss', reg_loss)

        return loss, probs

    def _build_train_op(self, loss, learning_rate):
        """
        Description:
            - Build a training operation minimizing the loss

        Args:
            - loss: symbolic loss

        Returns:
            - training operation
        """
        # adaptive learning rate
        opt = tf.train.AdamOptimizer(learning_rate)   

        # clip gradients by norm
        grads_params = opt.compute_gradients(loss) 
        clipped_grads_params = [(tf.clip_by_norm(
            g, self.flags.max_norm), p) 
            for (g, p) in grads_params]
        global_step = tf.Variable(0, trainable=False)
        train_op = opt.apply_gradients(
            clipped_grads_params, global_step=global_step)  

        # summaries
        # for (g, p) in clipped_grads_params:
        #     tf.histogram_summary('grads for {}'.format(p.name), g)

        return train_op

class FeedForwardNeuralNetwork(NeuralNetwork):
    def __init__(self, session, flags):
        """
        Description:
            - Initializes this network by storing the tf flags and 
                building the model.

        Args:
            - session: the session with which to execute model operations
            - flags: tensorflow flags object containing network options
        """
        super(FeedForwardNeuralNetwork, self).__init__(session, flags)

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
                shape=(None, self.flags.input_dim),
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
            - Builds a feed forward network with relu units.

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

        # build hidden layers
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

        # build output layer
        scores = tf.contrib.layers.fully_connected(hidden, 
                self.flags.output_dim, 
                activation_fn=None,
                weights_regularizer=weights_regularizer)

        # summaries
        # tf.histogram_summary('scores', scores)

        return scores

class WeightedFeedForwardNeuralNetwork(FeedForwardNeuralNetwork):

    def __init__(self, session, flags):
        super(WeightedFeedForwardNeuralNetwork, self).__init__(session, flags)

    def fit(self, dataset):
        """
        Description:
            - Fit this model to the provided dataset.

        Args:
            - dataset: the dataset to fit. Must implement
                the next_batch function  
        """
        self.start_time = time.time()

        # optionally load
        if self.flags.load_network:
            self.load()

        # fit the model to the dataset over a number of epochs
        for epoch in range(self.flags.num_epochs):
            train_loss, val_loss = 0, 0

            # train epoch
            for x, y, w in dataset.next_batch():
                outputs_list = [
                    self._summary_op, 
                    self._loss, 
                    self._losses,
                    self._train_op
                ]
                feed_dict = {
                    self._input_ph: x, 
                    self._target_ph: y,
                    self._dropout_ph: self.flags.dropout_keep_prob,
                    self._lr_ph: self.flags.learning_rate,
                    self._weights_ph: w
                }
                summary, loss, losses, _ = self.session.run(
                    outputs_list,
                    feed_dict=feed_dict)
                
                # update priorities as the negative of the (originally positive)
                # losses because priority dataset uses min heap
                dataset.update_priorities(-losses)
                self.train_writer.add_summary(summary, epoch)
                train_loss += loss

            # validation epoch
            for x, y in dataset.next_batch(validation=True):
                summary, loss = self.session.run([self._summary_op, self._loss],
                    feed_dict={self._input_ph: x, self._target_ph: y,
                    self._dropout_ph: 1., 
                    self._weights_ph: np.ones((len(x), 1)),
                    self._lr_ph: 0.})
                self.test_writer.add_summary(summary, epoch)
                val_loss += loss

            # print out progress if verbose
            if self.flags.verbose:
                self.log(epoch, dataset, train_loss, val_loss)

            # snapshot network
            self.save(epoch)

            # update hyperparameters
            self.update()

    def _build_model(self):
        """
        Description:
            - Builds the model, which entails defining placeholders, 
                a network, loss function, and train op. 

                Class variables created during this call all are assigned 
                to self in the body of this function, so everything that is 
                stored should be apparent from looking at this function.

                The results of these methods are passed in / out explicitly.
        """
        # placeholders
        (self._input_ph, self._target_ph, self._dropout_ph, 
            self._lr_ph, self._weights_ph) = self._build_placeholders()

        # network
        self._scores = self._build_network(
            self._input_ph, self._dropout_ph)

        # loss
        self._loss, self._probs, self._losses = self._build_loss(
            self._scores, self._target_ph, self._weights_ph)

        # train operation
        self._train_op = self._build_train_op(self._loss, self._lr_ph)

        # summaries
        self._summary_op = tf.summary.merge_all()

        # intialize the model
        self.session.run(tf.global_variables_initializer())

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
        input_ph, target_ph, dropout_ph, lr_ph = super(
            WeightedFeedForwardNeuralNetwork, self)._build_placeholders()
        weights_ph = tf.placeholder(tf.float32,
                shape=(None, 1),
                name="weights_ph")
        return input_ph, target_ph, dropout_ph, lr_ph, weights_ph

    def _build_loss(self, scores, targets, weights):
        """
        Description:
            - Build a loss function to optimize using the 
                scores of the network (unnormalized) and 
                the target values

        Args:
            - scores: unnormalized scores output from the network
                shape = (batch_size, output_dim)
            - targets: the target values
                shape = (batch_size, output_dim)

        Returns:
            - symbolic loss value
        """
        # create op for probability to use in 'predict'
        # even though this is only used for mse calculation
        probs = tf.sigmoid(scores)

        # create loss separately
        if self.flags.loss_type == 'ce':
            losses = tf.reduce_sum(
                tf.nn.sigmoid_cross_entropy_with_logits(scores, targets),
                reduction_indices=(1), keep_dims=True)
        elif self.flags.loss_type == 'mse':
            losses = tf.reduce_sum((probs - targets) ** 2, 
                reduction_indices=(1), keep_dims=True)
        else:
            raise(ValueError("invalid loss type: {}".format(
                self.flags.loss_type)))

        loss = tf.reduce_sum(weights * losses)

        # collect regularization losses
        reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        loss += reg_loss

        # summaries
        tf.histogram_summary('probs', probs)
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('l2 reg loss', reg_loss)

        return loss, probs, losses
