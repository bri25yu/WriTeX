import tensorflow as tf, numpy as np, cv2
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.python.keras.models import Sequential, save_model
from tensorflow.python.keras.optimizers import *
from tensorflow.python.keras.utils import to_categorical
import os

# Default training and testing data directories:
TRAIN_DIR, TEST_DIR = 'C:\\Users\\bri25\\Documents\\Python\\data\\', None

class Network:
    """
    Wrapper class for the CNN
    TRY: Normalize data to between [0, 1]
    Implement Adam optimizer
    """
    IMAGE_SIZE, TRAINING_TO_TESTING_RATIO = 45, 0.05
    LEARNING_RATE, MOMENTUM, DECAY = 0.001, 1.0, pow(10, -6)
    LOSS_FUNCTION = 'binary_crossentropy'
    temp_data_range = 2

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

    def append_data(self, data, labels, fname, name):
        """
        Method for how data is appended to later be passed into the network. 
        """
        data.append(np.array(cv2.imread(fname)))
        labels.append(name)

    def process_data(self):
        """
        This function takes the directories specified in the constructor and turns them into tf.datasets

        If the data set given only has one set of data, use 95% for training and the other 5% for testing/validation.
        If the data set only has one set of data, training is AUTOMATICALLY the first 95% of the data
        """
        symbol_names = os.listdir(self.train_dir)
        self.training_data, self.training_labels = [], []
        self.symbol_to_val = {}

        if not self.test_dir:
            for symbol_name in symbol_names:
            # for i in range(self.temp_data_range):
                # symbol_name = symbol_names[i]
                image_files = os.listdir(self.train_dir + symbol_name)
                for j in range(len(image_files)):
                    self.append_data(self.training_data, self.training_labels, self.train_dir + symbol_name + '\\' + image_files[j], i)
                    self.symbol_to_val[i] = symbol_name
            self.validation_data = None
        else:
            # ! TODO: fix lmao this definitely does NOT work
            testing_data, testing_labels = np.array([]), np.array([])
            for symbol_name in symbol_names:
                image_files = os.listdir(self.test_dir + symbol_name)
                for i in range(len(image_files)):
                    testing_data.append(np.array(cv2.imread(self.test_dir + symbol_name + '\\' + image_files[i])))
                    testing_labels.append(ord(symbol_name))
            self.validation_data = tuple([testing_data, testing_labels])

        self.training_labels = np.reshape(np.array(self.training_labels), (len(self.training_labels), 1))
        self.training_labels = to_categorical(self.training_labels, num_classes=self.temp_data_range)
        print("Finish process data")

    def get_layers(self):
        """
        Set the layers of the CNN as "logits" and "predictions"
        """
        FEATURES = [1, 32, 64, 128, 256]
        KERNEL = [6, 5, 4, 3, 3]
        LAYERS = len(KERNEL)

        self.model.add(Conv2D(FEATURES[0], KERNEL[0], input_shape = (self.IMAGE_SIZE, self.IMAGE_SIZE, 3), padding='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size = 2, strides = 2))
        
        for i in range(1, LAYERS):
            self.model.add(Conv2D(FEATURES[i], KERNEL[i], padding='same', activation='relu'))
            self.model.add(MaxPooling2D(pool_size = 2, strides = 2))

        self.model.add(Flatten())
        self.model.add(Dense(FEATURES[-1], activation = 'relu'))
        self.model.add(Dropout(0.1))
        self.model.add(Dense(2, activation='softmax'))

    def create_model(self):
        self.model = Sequential()
        self.get_layers()
        self.model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    
    def train_model(self, epochs = 10, batch_size = 128):
        """
        Call the model training function. 
        """
        if self.validation_data:
            self.model.fit(np.array(self.training_data), np.array(self.training_labels), epochs=epochs, 
                            batch_size=batch_size, shuffle=True, validation_data=self.validation_data)
        else:
            self.model.fit(np.array(self.training_data), np.array(self.training_labels), epochs=epochs, 
                            batch_size=batch_size, shuffle=True, validation_split=self.TRAINING_TO_TESTING_RATIO)
        save_model(self.model, 'data/model', overwrite=True, include_optimizer=True)

def main(TRAIN_DIR, TEST_DIR):
    my_network = Network(TRAIN_DIR, TEST_DIR)
    my_network.create_model()
    my_network.process_data()
    my_network.train_model(epochs = 20, batch_size = 1024)

if __name__ == '__main__':
    main(TRAIN_DIR, TEST_DIR)