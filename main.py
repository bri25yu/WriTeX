import numpy as np, tensorflow as tf
import glob, sys, os
import cv2
from recognition import *

def main():

    # Create and initialize LaTeX image to code mappingss
    # Open "notepad"
    
    train_dir, test_dir = 'C:\\Users\\bri25\\Documents\\Python\\data\\', None
    # Get training and testing data.
    features, labels, mode = [], [], 0

    recognition = Recognition(features, labels, mode)
    recognition.feed_data(process_data(train_dir, test_dir))

    classifier = tf.estimator.Estimator(model_fn = recognition.create_cnn)
    log = {"Probabilities" : "softmax result"}
    logging_hook = tf.train.LoggingTensorHook(tensors = log, every_n_iter = 50)

    print("Training input")
    # Train network using data
    train_input = tf.estimator.inputs.numpy_input_fn(
        x = {"x" : recognition.get_training_data()[0]},
        y = recognition.get_training_data()[1],
        batch_size = 500,
        num_epochs = None,
        shuffle = True,
        num_threads = 10
    )

    # Create test 
    test_input = tf.estimator.inputs.numpy_input_fn(
        x = {"x" : recognition.get_testing_data()[0]},
        y = recognition.get_testing_data()[1],
        num_epochs = 1,
        shuffle = False,
        num_threads = 10
    )

    print("Training network")
    # Classify and evaluate weights
    classifier.train(
        input_fn = recognition.train_cnn,
        steps = 1,
        hooks = [logging_hook]
    )

    print("Evaluating...")
    eval_input = tf.estimator.inputs.numpy_input_fn(
        x = {"x" : recognition.get_testing_data()[0]},
        y = recognition.get_testing_data()[1],
        batch_size = 500,
        num_epochs = 1,
        shuffle = False
    )

    #Output evaluation results
    evaluation = classifier.evaluate(input_fn = eval_input)
    print(evaluation)

    


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
    print("Begin process data")
    TESTING_TO_TRAINING_RATIO = 0.95
    symbol_names = os.listdir(train_dir)
    training_data, training_labels, testing_data, testing_labels = [], [], [], []

    if not test_dir:
        # for symbol_name in symbol_names:
        for i in range(2):
            symbol_name = symbol_names[i]
            print(symbol_name)
            image_files = os.listdir(train_dir + symbol_name)
            i = 0
            while i < int(TESTING_TO_TRAINING_RATIO*len(image_files)):
                training_data.append(np.array(cv2.imread(train_dir + symbol_name + '\\' + image_files[i])))
                training_labels.append(symbol_name)
                i+=1
            while i < len(image_files):
                testing_data.append(np.array(cv2.imread(train_dir + symbol_name + '\\' + image_files[i])))
                testing_labels.append(symbol_name)
                i+=1
    else:
        # Since this is a first iteration and we only have a singular set of data 
        #   (i.e. no official test data), it will automatically default to separating the 
        #   testing and training data as above. 
        pass
    print("Finish process data")
    return ((training_data, training_labels), (testing_data, testing_labels))
        
if __name__ == '__main__':
    main()