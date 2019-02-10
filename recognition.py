import tensorflow as tf, numpy as np

'''
Creates and trains the convolutional neural network for image recognition
'''
def create_cnn (features, labels, mode):
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
    logits = tf.layers.dense(inputs = rectify, units = 36)

    predictions = {
        "classes": tf.argmax(input = logits, axis = 1),
        "probabilities": tf.nn.softmax(logits, name = "softmax_tensor")
    }

    # For predicting provided image
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode = mode, prediction = predictions)
    
    # For training and back-propagating weights
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.001)
        train_op = optimizer.minimize(
            loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits), #also try just =loss
            global_step = tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode = mode, loss = loss, train_op = train_op)

    eval = {
        "Result accuracy" : tf.metrics.accuracy(labels = labels, predictions = predictions["classes"])
    }

    return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
    
# Trains and back-propagates weights given data

def get_data():

def train_cnn():
    ((training_data, (training_labels)), (eval_data, eval_labels)) = get_data()
    training_data /= np.float32(255)
    eval_data /= np.float32(255)

