""" Main file for running the CNN and RNN for handwriting recognition. """

# TensorFlow and Keras imports
import tensorflow as tf
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.python.keras.models import Sequential, save_model, load_model
from tensorflow.python.keras.optimizers import *

# Other library imports
import numpy as np, cv2
from multiprocessing.dummy import Pool as ThreadPool 
import os, argparse

# Custom library imports
from recognition_lib import *

class Network:
    """ Wrapper class for the CNN """
    
    def __init__(self):
        pass

    def train_model(self, 
                    model_dir, 
                    training_dir, 
                    testing_dir = None, 
                    batch_size = 1024, 
                    epochs = 10):
        my_model = self.CNN(model_dir=model_dir, training_dir=training_dir, testing_dir=testing_dir)
        my_model.process_data()
        my_model.create_model()
        my_model.train_model(batch_size=batch_size, epochs=epochs)

    def evaluate_model(self, model_dir, testing_dir):
        my_model = self.CNN(model_dir=model_dir, testing_dir=testing_dir)
        my_model.process_data()
        my_model.load_previous_model()
        my_model.evaluate_model()

    def predict(self, model_dir, predict_dir):
        my_model = self.CNN()

    class CNN:
        IMAGE_SIZE, TESTING_TO_TRAINING_RATIO = 45, 0.05
        NUM_THREADS = 10
        
        def __init__(self, model_dir, **kwargs):
            """ Set the training and testing data directories. 
            @param model_dir: Directory to load and save the model
            @param training_dir: Directory to get the training data from
            @param testing_dir: Directory to get the testing data from
            """
            self.model_dir = check_folder_style(model_dir)
            self.training_dir = check_folder_style(kwargs['training_dir']) if kwargs['training_dir'] else None
            self.testing_dir = check_folder_style(kwargs['testing_dir']) if kwargs['testing_dir'] else None

        # ----------------------------------------------------------------------------------
        # Actual methods
        def load_previous_model(self):
            self.model = load_model(self.model_dir + 'model')

        def train_model(self, epochs, batch_size):
            """ Will automatically evaluate model on testing data. """
            args = tuple((
                        np.array(self.training_data), 
                        np.array(self.training_labels)
                        ))
            kwargs = {
                        'epochs' : epochs, 
                        'batch_size' : batch_size, 
                        'shuffle' : True
                    }
            if self.testing_data:
                kwargs['validation_data'] = tuple((self.testing_data, self.testing_labels))
            else:
                kwargs['validation_split'] = self.TESTING_TO_TRAINING_RATIO
            self.model.fit(*args, **kwargs)
            self.model.save(self.model_dir)

        def evaluate_model(self):
            curr_loss, curr_acc = self.model.evaluate(
                                                        np.array(self.testing_data), 
                                                        np.array(self.testing_labels)
                                                    )
            print("Current loss: " + str(curr_loss), "Current accuracy: " + str(curr_acc))

        def create_model(self):
            self.model = Sequential()
            self.get_layers()
            self.model.compile(
                                optimizer='rmsprop',
                                loss='categorical_crossentropy',
                                metrics=['acc']
                            )

        def process_data(self):
            """ Takes training/testing data directories and turns them into tf.datasets """
            self.reset_data_labels()
            self.symbol_names = os.listdir(self.training_dir)
            self.num_classes = len(self.symbol_names)
            self.process_training_data()
            self.process_testing_data()

        # ----------------------------------------------------------------------------------
        # Helper methods
        def reset_data_labels(self):
            self.training_data, self.training_labels = [], []
            self.testing_data, self.testing_labels = [], []
            self.symbol_to_val = {}

        def process_symbol_directory(self, data, labels, direc, i):
            """ Helper function for processing all the loading of data symbols and training set data. """
            symbol_name = self.symbol_names[i]
            image_files = os.listdir(direc + symbol_name)
            for j in range(len(image_files)):
                append_data(data, labels, direc + symbol_name + '\\' + image_files[j], i)
                self.symbol_to_val[i] = symbol_name
            self.finished_loading += 1
            if self.finished_loading%10 == 0:
                print("Finished {0} / {1}".format(self.finished_loading, self.num_classes))

        def process_training_data(self):
            if self.training_dir:
                pool = ThreadPool(self.NUM_THREADS)
                self.finished_loading = 0
                args = get_pool_args(
                                    self.training_data, 
                                    self.training_labels, 
                                    self.training_dir, 
                                    self.num_classes
                                    )
                pool.starmap(self.process_symbol_directory, args)
                self.training_labels = categorize_labels(self.training_labels, self.num_classes)
                print("Finished loading training data")

        def process_testing_data(self):
            if self.testing_dir:
                pool = ThreadPool(self.NUM_THREADS)
                self.finished_loading = 0
                args = get_pool_args(
                                    self.testing_data, 
                                    self.testing_labels, 
                                    self.testing_dir, 
                                    self.num_classes
                                    )
                pool.starmap(self.process_symbol_directory, args)
                self.testing_labels = categorize_labels(self.testing_labels, self.num_classes)
                print("Finished loading testing data")

        def get_layers(self):
            """ Set the layers of the CNN. """
            FEATURES = [self.IMAGE_SIZE*self.IMAGE_SIZE, 32, 32, 32]
            KERNEL = [6, 5, 4, 3]
            LAYERS = len(KERNEL)

            self.model.add(Conv2D(FEATURES[0], KERNEL[0], input_shape = (self.IMAGE_SIZE, self.IMAGE_SIZE, 3), padding='same', activation='relu'))
            self.model.add(MaxPooling2D(pool_size = 2, strides = 2))
            for i in range(1, LAYERS):
                self.model.add(Conv2D(FEATURES[i], KERNEL[i], padding='same', activation='relu'))
                self.model.add(MaxPooling2D(pool_size = 2, strides = 2))
            self.model.add(Flatten())
            self.model.add(Dense(FEATURES[-1], activation = 'relu'))
            self.model.add(Dropout(0.1))
            self.model.add(Dense(self.num_classes, activation='softmax'))

def main():
    parser = get_parser()
    args = parser.parse_args()
    my_network = Network()

    model_dir = args.model_dir
    if args.train:
        training_dir = args.args_dir
        testing_dir = input("Please enter an additional testing data directory: \n")
        batch_size = int(input("Enter a batch_size. Default is 1024. \n"))
        epochs = int(input("Enter the number of training epochs. Default is 10. \n"))
        my_network.train_model(
                                model_dir, 
                                training_dir, 
                                testing_dir=testing_dir, 
                                batch_size=batch_size, 
                                epochs=epochs
                              )
    elif args.evaluate:
        testing_dir = args.args_dir
        my_network.evaluate_model(model_dir, testing_dir)
    elif args.predict:
        predict_dir = args.args_dir
        my_network.predict(model_dir, predict_dir)
    else:
        print("Please enter a valid combination of command line arguments")
    
if __name__ == '__main__':
    main()