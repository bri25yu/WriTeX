import tensorflow as tf, numpy as np, cv2
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizers import RMSprop
import os, sys

# Default training and testing data directories:
TRAIN_DIR, TEST_DIR = 'C:\\Users\\bri25\\Documents\\Python\\data\\', None

class Network:
    """
    Wrapper class for the CNN
    """

    def __init__(self, train_dir, test_dir=None):
        """
        Enable "eager execution"
        Set the training and testing data directories. 
        """
        # tf.enable_eager_execution()
        self.train_dir = train_dir + ("\\" if train_dir[-1] == "\\" else "")
        if test_dir:
            self.test_dir = test_dir + ("\\" if test_dir[-1] == "\\" else "")
        else: 
            self.test_dir = test_dir

    def process_data(self, batch_size = 500):
        """
        This function takes the directories specified in the constructor and turns them into tf.datasets

        If the data set given only has one set of data, use 95% for training and the other 5% for testing/validation.
        If the data set only has one set of data, training is AUTOMATICALLY the first 95% of the data
        """
        print("Begin process data")

        self.batch_size = batch_size

        TRAINING_TO_TESTING_RATIO = 0.95
        symbol_names = os.listdir(self.train_dir)
        self.training_data, self.training_labels, self.testing_data, self.testing_labels = [], [], [], []

        if not self.test_dir:
            # for symbol_name in symbol_names:
            for i in range(2):
                symbol_name = symbol_names[i]
                image_files = os.listdir(self.train_dir + symbol_name)
                i = 0
                while i < int(TRAINING_TO_TESTING_RATIO*len(image_files)):
                    self.training_data.append(np.array(cv2.imread(self.train_dir + symbol_name + '\\' + image_files[i])))
                    self.training_labels.append(symbol_name)
                    i+=1
                while i < len(image_files):
                    self.testing_data.append(np.array(cv2.imread(self.train_dir + symbol_name + '\\' + image_files[i])))
                    self.testing_labels.append(symbol_name)
                    i+=1
        else:
            # Since this is a first iteration and we only have a singular set of data 
            #   (i.e. no official test data), it will automatically default to separating the 
            #   testing and training data as above. 
            pass

        print("Finish process data")

    def get_layers(self):
        """
        Set the layers of the CNN as "logits" and "predictions"
        """
        IMAGE_SIZE = 28
        self.model.add(Conv2D(64, 5, input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3), padding="same", activation='relu'))
        self.model.add(MaxPooling2D(pool_size = 2, strides = 2))
        
        for i in range(1):
            self.model.add(Conv2D(64, 5, padding="same", activation='relu'))
            self.model.add(MaxPooling2D(pool_size = 2, strides = 2))

        self.model.add(Flatten())
        self.model.add(Dense(128, activation = 'relu'))
        self.model.add(Dropout(0.4))
        self.model.add(Dense(54, activation='softmax'))

    def create_model(self):
        self.model = Sequential()
        self.get_layers()
        self.model.compile(optimizer=RMSprop(lr=0.01),
              loss="categorical_crossentropy",
              metrics=["accuracy"])
    
    def set_testing_data(self, testing_data, testing_labels):
        """
        Sets the validation data and labels for model fitting as a tuple. 
        """
        self.validation_data = (testing_data, testing_labels)

    def train_model(self, epochs = 10, batch_size = 500):
        """
        Call the model training function. 
        """
        self.model.fit(self.training_data, self.training_labels, epochs=epochs, 
                        batch_size=batch_size, validation_data = self.validation_data)

if __name__ == '__main__':
    my_network = Network(TRAIN_DIR, TEST_DIR)
    my_network.create_model()
    my_network.process_data()
    my_network.train_model()