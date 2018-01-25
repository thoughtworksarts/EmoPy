import os, csv, cv2, datetime
from skimage import color, io
import numpy as np

class DataLoader:
    """
    Loads images and labels from dataset path.

    The *datapath* argument must be a path to a csv file or a directory containing the image and label data.

    If a csv_file is provided, user must indicate so by setting the *from_csv* argument to true and must provide values for the *csv_label_col* and *csv_image_col* arguments to indicate the indices of the images and labels in the csv file.

    If a directory path is provided, the directory must contain emotion label subdirectories containing all image samples pertaining to that label. If the directory contains image time series data, each emotion label subdirectory must contain time series subdirectories, each containing one time series sample. See the sample directories in the folder *examples/image_data*.

    :param from_csv: If true, images will be extracted from csv file.
    :param target_labels: List of target label values/strings.
    :param datapath: Location of image dataset.
    :param image_dimensions: Initial dimensions of raw training images.
    :param csv_label_col: Index of label value column in csv.
    :param csv_image_col: Index of image column in csv.
    :param time_steps: Number of images to load from each time series sample. Parameter must be provided to load time series data.
    """
    def __init__(self, from_csv=None, target_labels=None, datapath=None, image_dimensions=None, csv_label_col=None, csv_image_col=None, time_steps=None):
        self.from_csv = from_csv
        self.datapath = datapath
        self.image_dimensions = image_dimensions
        self.csv_label_col = csv_label_col
        self.csv_image_col = csv_image_col
        self.target_labels = target_labels
        self.time_steps = time_steps

        self._check_arguments()

    def get_data(self):
        """
        :return: List of images and list of corresponding labels from specified location. If loading from directory, also returns label index map (maps emotion label to integer value used during training).
        """
        if self.from_csv:
            return self._get_data_from_csv()
        elif self.time_steps is None:
            return self._get_data_from_directory()
        else:
            return self._get_image_series_data_from_directory()

    def _get_data_from_directory(self):
        """
        :return: List of images, list of corresponding labels, and label index map from specified directory.
        """
        images = list()
        labels = list()
        label_index_map = dict()
        label_directories = [dir for dir in os.listdir(self.datapath) if not dir.startswith('.')]
        for label_directory in label_directories:
            label_directory_path = self.datapath + '/' + label_directory
            image_files = [image_file for image_file in os.listdir(label_directory_path) if not image_file.startswith('.')]
            for image_file in image_files:
                image_file_path = label_directory_path + '/' + image_file
                image = io.imread(image_file_path)
                image = color.rgb2gray(image)
                images.append(image)

                if label_directory not in labels:
                    new_label_index = len(label_index_map.keys())
                    label_index_map[label_directory] = new_label_index
                labels.append(label_directory)

        label_values = list()
        label_count = len(label_index_map.keys())
        for label in labels:
            label_value = [0]*label_count
            label_value[label_index_map[label]] = 1.0
            label_values.append(label_value)

        self._check_data_not_empty(images)

        return np.array(images), np.array(label_values), label_index_map

    def _get_data_from_csv(self):
        """
        :return:  List of images and list of labels from csv file.
        """
        print('Extracting training data from csv...')
        start = datetime.datetime.now()

        images = list()
        labels = list()
        label_count = len(self.target_labels)
        label_map = dict()
        print('label_count: ' + str(label_count))
        with open(self.datapath) as csv_file:
            reader = csv.reader(csv_file, delimiter=',', quotechar='"')

            for row in reader:
                if row[self.csv_label_col] == 'emotion': continue
                raw_label = int(row[self.csv_label_col])
                if raw_label not in self.target_labels:
                    continue
                if raw_label not in label_map.keys():
                    label_map[raw_label] = len(label_map.keys())

                label = [0]*label_count
                label[label_map[raw_label]] = 1.0
                labels.append(np.array(label))

                image = np.asarray([int(pixel) for pixel in row[self.csv_image_col].split(' ')], dtype=np.uint8).reshape(self.image_dimensions)
                images.append(image)

        end = datetime.datetime.now()
        print('Training data extraction runtime - ' + str(end-start))

        self._check_data_not_empty(images)

        return np.array(images), np.array(labels)

    def _get_image_series_data_from_directory(self):
        """
        :return: List of image series data, list of corresponding labels, and label index map from specified directory.
        """
        image_series = list()
        labels = list()
        label_index_map = dict()
        label_directories = [dir for dir in os.listdir(self.datapath) if not dir.startswith('.')]
        for label_directory in label_directories:

            label_directory_path = self.datapath + '/' + label_directory
            series_directories = [series_directory for series_directory in os.listdir(label_directory_path) if not series_directory.startswith('.')]

            for series_directory in series_directories:
                series_directory_path = label_directory_path + '/' + series_directory

                self._check_series_directory_size(series_directory_path)

                new_image_series = list()
                image_files = [image_file for image_file in os.listdir(series_directory_path) if not image_file.startswith('.')]
                for image_file in image_files:
                    image_file_path = series_directory_path + '/' + image_file
                    image = io.imread(image_file_path)
                    image = color.rgb2gray(image)
                    new_image_series.append(image)

                start_idx = len(new_image_series) - self.time_steps
                end_idx = len(new_image_series)
                new_image_series = new_image_series[start_idx:end_idx]
                image_series.append(new_image_series)

                if label_directory not in labels:
                    new_label_index = len(label_index_map.keys())
                    label_index_map[label_directory] = new_label_index
                labels.append(label_directory)

        label_values = list()
        label_count = len(label_index_map.keys())
        for label in labels:
            label_value = [0]*label_count
            label_value[label_index_map[label]] = 1.0
            label_values.append(label_value)

        self._check_data_not_empty(image_series)

        return np.array(image_series), np.array(label_values), label_index_map

    def _check_arguments(self):
        if self.from_csv:
            self._check_csv_arguments()
        else:
            self._check_directory_arguments()

    def _check_csv_arguments(self):
        """
        Validates arguments for loading from csv file.
        """
        if self.csv_image_col is None or self.csv_label_col is None:
            raise ValueError('Must provide image and label indices to extract data from csv. csv_label_col and csv_image_col arguments not provided during DataLoader initialization.')

        if self.target_labels is None:
            raise ValueError('Must supply target_labels when loading data from csv.')

        if self.image_dimensions is None:
            raise ValueError('Must provide image dimensions when loading data from csv.')

        # check received valid csv file
        with open(self.datapath) as csv_file:

            # check image and label indices are valid
            reader = csv.reader(csv_file, delimiter=',', quotechar='"')
            num_cols = len(next(reader))
            if self.csv_image_col >= num_cols:
                raise(ValueError('Csv column index for image is out of range: %i' % self.csv_image_col))
            if self.csv_label_col >= num_cols:
                raise(ValueError('Csv column index for label is out of range: %i' % self.csv_label_col))

            # check image dimensions
            pixels = next(reader)[self.csv_image_col].split(' ')
            if len(pixels) != self.image_dimensions[0] * self.image_dimensions[1]:
                raise ValueError('Invalid image dimensions: %s' % str(self.image_dimensions))

    def _check_directory_arguments(self):
        """
        Validates arguments for loading from directories, including static image and time series directories.
        """
        if not os.path.isdir(self.datapath):
            raise (NotADirectoryError('Directory does not exist: %s' % self.datapath))
        if self.time_steps is not None:
            if self.time_steps < 1:
                raise ValueError('Time step argument must be greater than 0, but gave: %i' % self.time_steps)
            if not isinstance(self.time_steps, int):
                raise ValueError('Time step argument must be an integer, but gave: %s' % str(self.time_steps))

    def _check_data_not_empty(self, images):
        if len(images) == 0:
            raise AssertionError('csv file does not contain samples of specified labels: %s' % str(self.target_labels))

    def _check_series_directory_size(self, series_directory_path):
        image_files = [image_file for image_file in os.listdir(series_directory_path) if not image_file.startswith('.')]
        if len(image_files) < self.time_steps:
            raise ValueError('Time series sample found in path %s does not contain enough images for %s time steps.' % (series_directory_path, str(self.time_steps)))



