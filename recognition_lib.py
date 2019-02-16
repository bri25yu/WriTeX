""" Additional functions so recognition.py is a bit cleaner and more straightforward. """

# Library imports
import numpy as np, cv2
from tensorflow.python.keras.utils import to_categorical
import argparse

def check_folder_style(folder_name):
        """
        If a folder directory path does not have \\ at the end, add them in.
        This ensure no problems with file/directory loading in the future. 

        >>> s = check_folder_style("bob")
        >>> s
        'bob\\\\'
        >>> s = check_folder_style("")
        Traceback (most recent call last):
            ...
        AssertionError: Folder directory name cannot be empty
        >>> s = check_folder_style('bob\\\\')
        >>> s
        'bob\\\\'
        """
        assert folder_name, "Folder directory name cannot be empty"
        if folder_name[-1] == "\\":
            return folder_name
        return folder_name + "\\"

def append_data(data, labels, fname, name):
        """
        Add "fname" to list "data", and "name" to list "labels". 

        >>> append_data([], [], "fakeimage", "")
        Traceback (most recent call last):
            ...
        AssertionError: No image found at: fakeimage
        >>> import os, numpy as np, cv2
        >>> images_dir = 'images\\\\'
        >>> data, labels, images = [], [], os.listdir(images_dir)
        >>> append_data(data, labels, images_dir + images[0], 'bob')
        >>> len(data)
        1
        >>> labels[0]
        'bob'
        """
        image = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
        assert image is not None, "No image found at: " + fname
        data.append(np.array(image))
        labels.append(name)

def categorize_labels(labels, num_classes):
        """
        Turn all the current labels into numpy arrays for NN processing. 

        >>> categorize_labels([], 5)
        Traceback (most recent call last):
            ...
        AssertionError: Labels to categorize cannot be empty
        >>> categorize_labels([1], 0)
        Traceback (most recent call last):
            ...
        AssertionError: The number of categorization classes must be greater than 0
        >>> categorize_labels([0, 1, 2, 1], 3)
        array([[1., 0., 0.],
               [0., 1., 0.],
               [0., 0., 1.],
               [0., 1., 0.]], dtype=float32)
        """
        assert len(labels), "Labels to categorize cannot be empty"
        assert num_classes>0, "The number of categorization classes must be greater than 0"
        labels = np.reshape(np.array(labels), (len(labels), 1))
        return to_categorical(labels, num_classes=num_classes)

def add_then_remove(from_arr, to_arr, index):
    """
    Removes the value at index from from_arr and appends it to the end of to_arr 

    >>> from_arr, to_arr = [0, 1, 2, 3], []
    >>> add_then_remove(from_arr, to_arr, 2) 
    >>> from_arr
    [0, 1, 3]
    >>> to_arr
    [2]
    """
    to_arr.append(from_arr[index])
    from_arr.pop(index)

def get_pool_args(data, labels, from_dir, num_classes):
        """ Returns an array of the set of distrubuted data processing for Thread pooling. """
        return [tuple([data, labels, from_dir, i]) for i in range(num_classes)]

def get_parser():
    parser = argparse.ArgumentParser(description="Utilize the CNN")
    parser.add_argument("--train", help="Train the CNN", action="store_true")
    parser.add_argument("--evaluate", help="Evaluate the CNN", action="store_true")
    parser.add_argument("--predict", help="Use the CNN to predict some values", action="store_true")
    parser.add_argument("model_dir")
    parser.add_argument("args_dir")
    return parser

    