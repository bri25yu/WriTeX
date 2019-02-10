import tensorflow as tf, numpy as np

class Recognition:
    '''
    Convolutional Neural Network (CNN) object wrapper class.
    Creates and trains the convolutional neural network for image recognition
    '''

    def __init__ (self, features, labels, mode):
        create_cnn(features, labels, mode)

    def feed_data(self, data):
        """
        Input data must be a second-level nested tuple
        Refer to process_data() in "main.py" for proper format
        """
        assert type(data) == tuple, "Data is not a tuple"
        assert type(data[0]) == tuple, "Training data is not a tuple"
        assert type(data[1]) == tuple, "Testing data is not a tuple"

        self.training_data, self.training_labels = data[0][0], data[0][1]
        self.testing_data, self.testing_labels = data[1][0], data[1, 1]

    def get_training_data (self):
        """
        Returns a tuple of (training_data, training_labels)
        """
        return (self.training_data, self.training_labels)
    
    def get_testing_data (self):
        """
        Returns a tuple of (testing_data, testing_labels)
        """
        return (self.testing_data, self.testing_labels)

    def create_cnn (self, features, labels, mode):
        self.labels, self.mode = labels, mode
        # Initializes input layer
        input_layer = tf.reshape(features["x"], [-1, 28, 28])

        # Initializes first convolutional layer
        layer_1 = tf.layers.conv2d(
            inputs = input_layer, 
            filters = 32,
            kernel_size = [5, 5],
            padding = "same",
            activation = tf.nn.relu)
        
        # Initializes first max pooling layer
        max_pool_1 = tf.layers.max_pooling2d(inputs = layer_1, pool_size = [2, 2], strides = 2)
        
        # Initializes second convolutional layer
        layer_2 = tf.layers.conv2d(
            inputs = max_pool_1,
            filters = 64,
            kernel_size = [5, 5],
            padding = "same",
            activation = tf.nn.relu)

        # Initializes second max pooling layer
        max_pool_2 = tf.layers.max_pooling2d(inputs = layer_2, pool_size = [2, 2], strides = 2)

        # Initializes third convolutional layer
        layer_3 = tf.layers.conv2d(
            inputs = max_pool_2,
            filters = 128,
            kernel_size = [5, 5],
            padding = "same",
            activation = tf.nn.relu
        )

        # Initializes third max pooling layer
        max_pool_3 = tf.layers.max_pooling2d(inputs = layer_3, pool_size = [2, 2], strides = 2)

        # Projects onto dense layer
        flatten_pool_3 = tf.reshape(max_pool_3, [-1, max_pool_3.size])
        dense_layer = tf.layers.dense(inputs = flatten_pool_3, units = 1024, activation = tf.nn.relu)
        rectify = tf.layers.dropout(inputs = dense_layer, rate = 0.4, training = mode == tf.estimator.ModeKeys.TRAIN)

        # Creates logits and final output probability layer
        self.logits = tf.layers.dense(inputs = rectify, units = 36)

        self.predictions = {
            "classes": tf.argmax(input = logits, axis = 1),
            "probabilities": tf.nn.softmax(logits, name = "softmax_tensor")
        }
        
    def predict(self):
        # For predicting provided image

        if self.mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode = self.mode, prediction = self.predictions)
        eval_metric_ops = {
            "Result accuracy" : tf.metrics.accuracy(labels = self.labels, 
            predictions = self.predictions["classes"])
            }

        return tf.estimator.EstimatorSpec(mode=self.mode, loss=loss, eval_metric_ops=eval_metric_ops)

    def train_cnn(self):
        # Trains and back-propagates weights given data
        self.training_data /= np.float32(255)
        self.eval_data /= np.float32(255)
        
        # For training and back-propagating weights
        if self.mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.001)
            train_op = optimizer.minimize(
                loss = tf.losses.sparse_softmax_cross_entropy(labels=self.labels, logits=self.logits), #also try just =loss
                global_step = tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode = self.mode, loss = loss, train_op = train_op)
