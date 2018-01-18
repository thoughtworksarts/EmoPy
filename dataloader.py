import os, csv, cv2, datetime
from skimage import color, io
import numpy as np


EMOTION_DIMENSION_COUNT = 4 # emotional dimensions: arousal, valence, expectation, power

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
    def __init__(self, from_csv=None, target_labels=None, datapath=None, image_dimensions=None, csv_label_col=None, csv_image_col=None, avep=False):
        self.from_csv = from_csv
        self.datapath = datapath
        self.image_dimensions = image_dimensions
        self.csv_label_col = csv_label_col
        self.csv_image_col = csv_image_col
        self.target_labels = target_labels
        self.avep = avep

    def get_data(self):
        """
        :return: list of images and list of corresponding labels from specified location
        """
        if self.from_csv:
            return self._get_data_from_csv()
        elif not self.avep:
            return self._get_data_from_directory()
        else:
            images = self._get_image_series_from_directory()
            labels = self._get_image_series_labels()
            return images, labels

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

    # --------------- Image Series Loading ---------------- #

    def _get_image_series_from_directory(self):
        """
        :return:  list of images from directory location, resized to specified target dimensions
        """
        images = list()
        for sub_directory in os.listdir(self.datapath):
            if not sub_directory.startswith('.'):
                sub_directory_path = self.datapath + '/' + sub_directory
                for image_file in os.listdir(sub_directory_path):
                    if not image_file.startswith('.'):
                        image_file_path = sub_directory_path + '/' + image_file
                        image = io.imread(image_file_path)
                        image = color.rgb2gray(image)
                        images.append(image)
        return np.array(images)

    def _get_image_series_labels(self):
        raw_labels = self._get_raw_image_series_labels()
        training_label_array = list()
        for time_series_key in raw_labels:
            time_series = raw_labels[time_series_key]
            training_label_array += time_series

        return np.array(training_label_array)

    def _get_raw_image_series_labels(self):
        """
        Uses 20 photo series from the Cohn-Kanade dataset
        manually labeled by AP
        arousal(least, most), valence(negative, positive), power, anticipation
        """
        raw_labels = {1: [10, [.6, .4, .7, .6], [.9, .1, .8, .9]],
                               2: [9, [.2, .5, .6, .1], [.3, .4, .5, .2]],
                               3: [10, [.8, .9, .2, .9], [.99, .99, .1, .99]],
                               4: [10, [.2, .4, .4, .5], [.8, .2, .7, .6]],
                               5: [8, [.2, .4, .2, .1], [.5, .5, .5, .5]],
                               6: [10, [.8, .2, .2, .5], [.9, .1, .1, .5]],
                               7: [10, [.7, .4, .5, .6], [.8, .2, .8, .7]],
                               8: [9, [.5, .5, .4, .5], [.6, .4, .5, .3]],
                               9: [10, [.6, .4, .4, .7], [.9, .1, .1, .9]],
                               10: [10, [.1, .5, .2, .1], [.7, .2, .2, .5]],
                               11: [10, [.2, .4, .5, .2], [.3, .5, .4, .2]],
                               12: [10, [.6, .2, .2, .4], [.8, .1, .1, .4]],
                               13: [10, [.6, .4, .7, .5], [.8, .2, .8, .5]],
                               14: [9, [.1, .5, .5, .5], [.1, .4, .5, .4]],
                               15: [10, [.1, .4, .5, .5], [.8, .3, .4, .9]],
                               16: [10, [.6, .4, .7, .6], [.7, .2, .2, .5]],
                               17: [10, [.5, .4, .6, .5], [.7, .2, .7, .6]],
                               18: [10, [.7, .4, .2, .7], [.9, .1, .1, .9]],
                               19: [10, [.7, .3, .3, .5], [.8, .1, .1, .6]],
                               20: [10, [.6, .3, .2, .5], [.9, .1, .1, .6]]}
        labels = dict()
        for time_series_key in raw_labels:
            time_series = raw_labels[time_series_key]
            num_images = time_series[0]
            increment = [(time_series[2][emotion_dimension_idx] - time_series[1][emotion_dimension_idx]) / num_images for emotion_dimension_idx in range(EMOTION_DIMENSION_COUNT)]
            labels[time_series_key] = [[increment[label_idx]*image_idx + (time_series[1][label_idx]) for label_idx in range(EMOTION_DIMENSION_COUNT)] for image_idx in range(num_images)]

        return labels

    def _get_time_delay_data(self, datapoints, labels, time_delay=2):
        """
        Adds time delay dimension to given datapoints.

        :param datapoints: list of datapoints
        :param labels: list of labels
        :param time_delay: number of previous time steps to add to each datapoint
        :return: list of time-delayed datapoints
        """
        features = list()
        for data_point_idx in range(time_delay, len(datapoints)):
            data_point = [datapoints[data_point_idx - offset] for offset in range(time_delay + 1)]
            features.append([data_point])

        labels = labels[time_delay:len(labels)]

        return (features, labels)