import tensorflow as tf, numpy as np, cv2
from tensorflow.keras import layers

class Network:
    """
    Wrapper class for the CNN
    """

    def __init__(self):
        """
        Empty constructor for this wrapper class!
        """
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

    def create_model(self):
        self.model = tf.keras.Sequential([
            # Adds a densely-connected layer with 64 units to the model:
            layers.Dense(64, activation='relu', input_shape=(32,)),
            # Add another:
            layers.Dense(64, activation='relu'),
            # Add a softmax layer with 10 output units:
            layers.Dense(10, activation='softmax')
            ])

        self.model.compile(optimizer=tf.train.RMSPropOptimizer(0.01),
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=[tf.keras.metrics.categorical_accuracy])

    def set_training_data(self, training_data, training_labels):
        """
        Sets the object's training data and training labels. 
        """
        self.training_data = training_data
        self.training_labels = training_labels
    
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
                        batch_size=batch_size, validation_data = self.validation_data
                        )

if __name__ == '__main__':
    my_network = Network()
    my_network.create_model()

    data = np.random.random((1000, 32))
    labels = np.random.random((1000, 10))

    val_data = np.random.random((100, 32))
    val_labels = np.random.random((100, 10))

    my_network.set_training_data(data, labels)
    my_network.set_testing_data(val_data, val_labels)

    my_network.train_model(batch_size=32)