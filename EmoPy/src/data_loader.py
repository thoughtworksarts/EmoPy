import csv, cv2
import numpy as np
from sklearn.model_selection import train_test_split

from EmoPy.src.dataset import Dataset


class _DataLoader(object):
    """
    Abstract class to load image and label data from a directory or csv file.

    Methods load_data and _validate_arguments must be implemented by subclasses.
    """
    def __init__(self, validation_split, time_delay=None):
        self.validation_split = validation_split
        self.time_delay = time_delay
        self._validate_arguments()

    def load_data(self):
        """
        Loads image and label data from path specified in subclass initialization.

        :return: Dataset object containing image and label data.
        """
        raise NotImplementedError("Class %s doesn't implement load_data()" % self.__class__.__name__)

    def _load_dataset(self, images, labels, emotion_index_map):
        """
        Loads Dataset object with images, labels, and other data.

        :param images: numpy array of image data
        :param labels: numpy array of one-hot vector labels
        :param emotion_index_map: map linking string/integer emotion class to integer index used in labels vectors

        :return: Dataset object containing image and label data.
        """
        train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=self.validation_split, random_state=42, stratify=labels)
        dataset = Dataset(train_images, test_images, train_labels, test_labels, emotion_index_map, self.time_delay)
        return dataset

    def _validate_arguments(self):
        if self.out_channels not in (1, 3):
            raise ValueError("Out put channel should be either 3(RGB) or 1(Grey) but got {channels}".format(channels=self.out_channels))
        if self.validation_split < 0 or self.validation_split > 1:
            raise ValueError("validation_split must be a float between 0 and 1")
        raise NotImplementedError("Class %s doesn't implement _validate_arguments()" % self.__class__.__name__)

    def _reshape(self, image):
        if image.ndim == 2:
            image = np.expand_dims(image, axis=2)
        channels = image.shape[-1]

        if channels == 3 and self.out_channels == 1:
            gray = cv2.cvtColor(image, code=cv2.COLOR_BGR2GRAY)
            return np.expand_dims(gray, axis=2)
        if channels == 1 and self.out_channels == 3:
            return np.repeat(image, repeats=3, axis=2)
        return image

    def _check_data_not_empty(self, images):
        if len(images) == 0:
            raise AssertionError('csv file does not contain samples of specified labels: %s' % str(self.label_map.keys()))

    def _vectorize_labels(self, label_index_map, labels):
        label_values = list()
        label_count = len(label_index_map.keys())
        for label in labels:
            label_value = [0] * label_count
            label_value[label_index_map[label]] = 1.0
            label_values.append(label_value)
        return label_values
