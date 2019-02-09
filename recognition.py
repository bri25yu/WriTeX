import pylon, tensorflow as tf

'''
Text recognition using Convolutional Neural Networks and Tensorflow
'''

class Recognition(object):

    def __init__(self):
        #Setting default batch size and image dimension values
        self.batch_size = -1
        self.image_height = 28
        self.image_width = 28

    def create_cnn(self, features, labels, mode):
        #Initializing input layer
        input_layer = tf.reshape(features["x"], [self.batch_size, self.image_height, self.image_width])

        #Initializing 1st convolutional layer
        layer_1 = tf.layers.conv2d(
            inputs = input_layer,
            filters = 32,
            kernel_size = [5, 5],
            padding = "same",
            activation = tf.nn.relu
        )

        #Initializing 1st pooling layer
        pool_1 = tf.layers.max_pooling2d(inputs = layer_1, pool_size = [2, 2], strides = 2)

        #Initializng 2nd convolutional layer
        layer_2 = tf.layers.conv2d(
            inputs = pool_1,
            filters = 64,
            kernel_size = [5, 5],k
            padding = "same",
            activation = tf.nn.relu
        )

        #Initializing 2nd pooling layer
        pool_2 = tf.layers.max_pooling2d(inputs = layer_2, pool_size = [2, 2], strides = 2)
        flatten_pool2 = tf.reshape(pool_2, [-1, pool_2.size])



