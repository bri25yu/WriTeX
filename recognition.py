import tensorflow as tf, numpy as np, cv2
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.python.keras.models import Sequential, save_model, load_model
from tensorflow.python.keras.optimizers import *
from tensorflow.python.keras.utils import to_categorical
from multiprocessing.dummy import Pool as ThreadPool 
import os

# Default training and testing data directories:
TRAIN_DIR, TEST_DIR = 'C:\\Users\\bri25\\Documents\\Python\\data\\', None
SAVE_DIR = 'data/'

class Network:
    """
    Wrapper class for the CNN
    TRY: Normalize data to between [0, 1]
    Implement Adam optimizer
    """
    IMAGE_SIZE, TRAINING_TO_TESTING_RATIO = 45, 0.95
    NUM_THREADS = 100

    def __init__(self, train_dir, save_dir, test_dir=None):
        """
        Set the training and testing data directories. 
        """
        # tf.enable_eager_execution()
        self.train_dir = train_dir + ("\\" if train_dir[-1] == "\\" else "")
        self.save_dir = save_dir
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

    def reset_data_labels(self):
        """
        Initializes all the required current data storage devices. 
        """
        self.training_data, self.training_labels = [], []
        self.testing_data, self.testing_labels = [], []
        self.symbol_to_val = {}
        self.finished_loading = 0

    def process_symbol_directory(self, data, labels, direc, symbol_names, i):
        """
        Helper function for processing all the loading of data symbols and training set data. 
        """
        symbol_name = symbol_names[i]
        image_files = os.listdir(direc + symbol_name)
        for j in range(len(image_files)):
            self.append_data(data, labels, direc + symbol_name + '\\' + image_files[j], i)
            self.symbol_to_val[i] = symbol_name
        self.finished_loading += 1
        if self.finished_loading%10 == 0:
            print("Finished {0} / {1}".format(self.finished_loading, self.num_classes))

    def get_test_from_training(self):
        random_ints = np.random.randint(len(self.training_labels), size=(int((1-self.TRAINING_TO_TESTING_RATIO)*len(self.training_labels)), 1))
        random_ints = np.sort(random_ints, 0)
        for i in range(len(random_ints)):
            val = random_ints[len(random_ints)-i-1][0]
            self.testing_data.append(self.training_data[val])
            self.testing_labels.append(self.training_labels[val])
            self.training_data.pop(val)
            self.training_labels.pop(val)
        print("Finished selecting test data")

    def categorize_labels(self):
        """
        Turn all the current labels into numpy arrays for NN processing. 
        """
        self.training_labels = np.reshape(np.array(self.training_labels), (len(self.training_labels), 1))
        self.training_labels = to_categorical(self.training_labels, num_classes=self.num_classes)
        self.testing_labels = np.reshape(np.array(self.testing_labels), (len(self.testing_labels), 1))
        self.testing_labels = to_categorical(self.testing_labels, num_classes=self.num_classes)

    def get_pool_args(self, symbol_names):
        """
        Returns an array of the set of distrubuted data processing for Thread pooling.
        This is the function for no test directory.
        """
        args = []
        for i in range(self.num_classes):
            args.append(tuple([self.training_data, self.training_labels, self.train_dir, symbol_names, i]))
        return args

    def process_data(self):
        """
        This function takes the directories specified in the constructor and turns them into tf.datasets

        If the data set given only has one set of data, use 95% for training and the other 5% for testing/validation.
        If the data set only has one set of data, training is AUTOMATICALLY the first 95% of the data
        """
        self.reset_data_labels()
        symbol_names = os.listdir(self.train_dir)
        self.num_classes = len(symbol_names)

        pool = ThreadPool(self.NUM_THREADS)
        if not self.test_dir:
            args = self.get_pool_args(symbol_names)
            pool.starmap(self.process_symbol_directory, args)
            print("Finished loading data")
            self.get_test_from_training()
            self.categorize_labels()           
        else:
            # ! TODO: fix lmao this definitely does NOT work
            testing_data, testing_labels = np.array([]), np.array([])
            for symbol_name in symbol_names:
                image_files = os.listdir(self.test_dir + symbol_name)
                for i in range(len(image_files)):
                    testing_data.append(np.array(cv2.imread(self.test_dir + symbol_name + '\\' + image_files[i])))
                    testing_labels.append(ord(symbol_name))
        print("Finished process data")

    def get_layers(self):
        """
        Set the layers of the CNN. 
        """
        FEATURES, KERNEL = [1, 32, 64, 128, 256], [6, 5, 4, 3, 3]
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
        past_best_model = load_model(self.save_dir + 'model')
        best_loss, best_acc = past_best_model.evaluate(np.array(self.testing_data), np.array(self.testing_labels))
        self.model.fit(np.array(self.training_data), np.array(self.training_labels), epochs=epochs, 
                        batch_size=batch_size, shuffle=True)
        curr_loss, curr_acc = self.model.evaluate(np.array(self.testing_data), np.array(self.testing_labels))
        if curr_loss * curr_acc >= best_loss * best_acc:
            print("Best run yet!")
            save_model(self.model, self.save_dir + 'model', overwrite=True, include_optimizer=True)
            best_loss, best_acc = curr_loss, curr_acc

    def evaluate_model(self, testing_data, testing_labels):
        # TODO: Finish later!
        pass

    def implement(self):
        print("Cnn success")

def main(TRAIN_DIR, 
         SAVE_DIR, 
         TEST_DIR, 
         train = False,
         implementing = False,
         controller = None,   
         epochs = 10, 
         batch_size = 1024, 
         testing_data = None, 
         testing_labels = None):
    """
    """
    my_network = Network(TRAIN_DIR, SAVE_DIR, TEST_DIR)
    my_network.process_data()
    my_network.create_model()
    if train:
        my_network.train_model(epochs = epochs, batch_size = batch_size)
    elif implementing:
        my_network.load_model()
        controller.set_CNN(my_network)
    else:
        my_network.evaluate_model(epochs, batch_size, testing_data, testing_labels)

if __name__ == '__main__':
    main(TRAIN_DIR, SAVE_DIR, TEST_DIR)