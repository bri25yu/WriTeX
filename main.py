import numpy as np, tensorflow as tf
import glob, sys, os
import cv2
from recognition import *

def main():

    # Create and initialize LaTeX image to code mappingss
    # Open "notepad"
    
    train_dir, test_dir = 'C:\\Users\\bri25\\Documents\\Python\\data\\', None
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
    If the data set only has one set of data, training is AUTOMATICALLY the first 95% of the data
    """

    symbol_names = os.listdir(train_dir)
    training_data, training_labels, testing_data, testing_labels = [], [], [], []

    if not test_dir:
        for symbol_name in symbol_names:
            image_files = os.listdir(train_dir + symbol_name)
            i = 0
            while i < int(0.95*len(image_files)):
                training_data.append(cv2.imread(train_dir + symbol_name + '\\' + image_files[i]))
                training_labels.append(symbol_name)
                i+=1
            while i < len(image_files):
                testing_data.append(cv2.imread(train_dir + symbol_name + '\\' + image_files[i]))
                testing_labels.append(symbol_name)
                i+=1
    else:
        # Since this is a first iteration and we only have a singular set of data 
        #   (i.e. no official test data), it will automatically default to separating the 
        #   testing and training data as above. 
        pass
    return ((training_data, training_labels), (testing_data, testing_labels))
        
if __name__ == '__main__':
    main()