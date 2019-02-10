import tensorflow as tf, numpy as np, cv2
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizers import RMSprop
import os

# Default training and testing data directories:
TRAIN_DIR, TEST_DIR = 'C:\\Users\\bri25\\Documents\\Python\\data\\', None

class Network:
    """
    Wrapper class for the CNN
    """
    IMAGE_SIZE, TRAINING_TO_TESTING_RATIO = 45, 0.95

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
        self.batch_size = batch_size
        symbol_names = os.listdir(self.train_dir)
        self.training_data, self.training_labels = [], []

        if not self.test_dir:
            # for symbol_name in symbol_names:
            for i in range(2):
                symbol_name = symbol_names[i]
                image_files = os.listdir(self.train_dir + symbol_name)
                for i in range(len(image_files)):
                    self.training_data.append(np.array(cv2.imread(self.train_dir + symbol_name + '\\' + image_files[i])))
                    self.training_labels.append(ord(symbol_name))
            self.validation_data = None
        else:
            testing_data, testing_labels = [], []
            for symbol_name in symbol_names:
                image_files = os.listdir(self.test_dir + symbol_name)
                for i in range(len(image_files)):
                    testing_data.append(np.array(cv2.imread(self.test_dir + symbol_name + '\\' + image_files[i])))
                    testing_labels.append(ord(symbol_name))
            self.validation_data = tuple([testing_data, testing_labels])
        print("Finish process data")

    def get_layers(self):
        """
        Set the layers of the CNN as "logits" and "predictions"
        """
        self.model.add(Conv2D(64, 5, input_shape = (self.IMAGE_SIZE, self.IMAGE_SIZE, 3), padding="same", activation='relu'))
        self.model.add(MaxPooling2D(pool_size = 2, strides = 2))
        
        for i in range(1):
            self.model.add(Conv2D(64, 5, padding="same", activation='relu'))
            self.model.add(MaxPooling2D(pool_size = 2, strides = 2))

        self.model.add(Flatten())
        self.model.add(Dense(128, activation = 'relu'))
        self.model.add(Dropout(0.4))
        self.model.add(Dense(1, activation='softmax'))

    def create_model(self):
        self.model = Sequential()
        self.get_layers()
        self.model.compile(optimizer=RMSprop(lr=0.01),
              loss="binary_crossentropy",
              metrics=["acc"])
    
    def train_model(self, epochs = 10, batch_size = 500):
        """
        Call the model training function. 
        """
        if self.validation_data:
            self.model.fit(np.array(self.training_data), np.array(self.training_labels), epochs=epochs, 
                            batch_size=batch_size, validation_data=self.validation_data)
        else:
            self.model.fit(np.array(self.training_data), np.array(self.training_labels), epochs=epochs, 
                            batch_size=batch_size, validation_split=self.TRAINING_TO_TESTING_RATIO)

def main(TRAIN_DIR, TEST_DIR):
    my_network = Network(TRAIN_DIR, TEST_DIR)
    my_network.create_model()
    my_network.process_data()
    my_network.train_model()

if __name__ == '__main__':
    main(TRAIN_DIR, TEST_DIR)