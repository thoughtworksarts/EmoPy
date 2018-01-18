import os, csv, cv2, datetime
from skimage import color, io
import numpy as np

class DataLoader:
    """
    Loads images and labels from dataset path.

    :param from_csv: if true, images will be extracted from csv file
    :param target_labels: list of target label values/strings
    :param datapath: location of image dataset
    :param image_dimensions: initial dimensions of raw training images
    :param csv_label_col: index of label value column in csv
    :param csv_image_col: index of image column in csv
    """
    def __init__(self, from_csv=None, target_labels=None, datapath=None, image_dimensions=None, csv_label_col=None, csv_image_col=None, time_steps=None):
        self.from_csv = from_csv
        self.datapath = datapath
        self.image_dimensions = image_dimensions
        self.csv_label_col = csv_label_col
        self.csv_image_col = csv_image_col
        self.target_labels = target_labels
        self.time_steps = time_steps

    def get_data(self):
        """
        :return: list of images and list of corresponding labels from specified location
        """
        if self.from_csv:
            return self._get_data_from_csv()
        elif self.time_steps is None:
            return self._get_data_from_directory()
        else:
            return self._get_image_series_data_from_directory()
            # images = self._get_image_series_from_directory()
            # labels = self._get_image_series_labels()
            # return images, labels

    def _get_data_from_directory(self):
        """
        :return: list of images and list of corresponding labels from specified directory
        """
        images = list()
        labels = list()
        label_map = dict()
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
                    new_label_index = len(label_map.keys())
                    label_map[label_directory] = new_label_index
                labels.append(label_directory)

        label_values = list()
        label_count = len(label_map.keys())
        for label in labels:
            label_value = [0]*label_count
            label_value[label_map[label]] = 1.0
            label_values.append(label_value)

        return np.array(images), np.array(label_values)

    def _get_data_from_csv(self):
        """
        :return:  list of images and list of labels from csv file
        """
        print('Extracting training data from csv...')
        start = datetime.datetime.now()

        images = list()
        labels = list()
        with open(self.datapath) as csv_file:
            reader = csv.reader(csv_file, delimiter=',', quotechar='"')

            for row in reader:
                if row[self.csv_label_col] == 'emotion': continue
                if int(row[self.csv_label_col]) not in self.target_labels:
                    continue

                label = [0]*7
                label[int(row[self.csv_label_col])] = 1.0
                labels.append(np.array(label))

                image = np.asarray([int(pixel) for pixel in row[self.csv_image_col].split(' ')], dtype=np.uint8).reshape(self.image_dimensions)
                images.append(image)

        end = datetime.datetime.now()
        print('Training data extraction runtime - ' + str(end-start))

        return np.array(images), np.array(labels)

    def _get_image_series_data_from_directory(self):
        """
        :return: list of image series data and list of corresponding labels from specified directory
        """
        image_series = list()
        labels = list()
        label_map = dict()
        label_directories = [dir for dir in os.listdir(self.datapath) if not dir.startswith('.')]
        for label_directory in label_directories:
            label_directory_path = self.datapath + '/' + label_directory
            series_directories = [series_directory for series_directory in os.listdir(label_directory_path) if not series_directory.startswith('.')]

            for series_directory in series_directories:
                series_directory_path = label_directory_path + '/' + series_directory
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
                    new_label_index = len(label_map.keys())
                    label_map[label_directory] = new_label_index
                labels.append(label_directory)

        label_values = list()
        label_count = len(label_map.keys())
        for label in labels:
            label_value = [0]*label_count
            label_value[label_map[label]] = 1.0
            label_values.append(label_value)

        return np.array(image_series), np.array(label_values)