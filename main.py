import numpy as np, tensorflow as tf, glob, sys
from recognition import *

def main():

    # Create and initialize LaTeX image to code mappingss
    # Open "notepad"
    
    train_dir, test_dir = 'C:\\Users\\bri25\\Documents\\Python\\data', None
    # Get training and testing data.

    classifier = tf.estimator.Estimator(model_fn = create_cnn, model_dir = "/tmp/mnist_convnet_model")
    log = {"Probabilities" : "softmax result"}
    logging_hook = tf.train.LoggingTensorHook(tensors = log, every_n_iter = 50)

    while False: 
        # while the user is running this program

        # user writes something
        # 2-3 second delay for the user to finish what they're writing
        # convert user writing into latex code
        # give user to the view pdf


        #user has option to resize things
        pass

def process_data(train_dir, test_dir = None):
    """
    If the data set given only has one set of data, use 95% for training and the other 5% for testing/validation.
    """
    if not test_dir:
        

if __name__ == '__main__':
    main()