import numpy as np, tensorflow as tf
from recognition import *


def main():
    """
    Main method for
    """

    # Create and initialize LaTex image to code mappingss
    # Open "notepad"
    # Change model_dir to where the model data is stored
    classifier = tf.estimator.Estimator(model_fn = create_cnn, model_dir = "/tmp/mnist_convnet_model")
    log = {"Probabilities" : "softmax result"}
    logging_hook = tf.train.LoggingTensorHook(tensors = log, every_n_iter = 50)

    while False: 
        # while the user is running this progra

        # user writes something
        # 2-3 second delay for the user to finish what they're writing
        # convert user writing into latex code
        # give user to the view pdf


        #user has option to resize things
        pass



if __name__ == '__main__':
    main()