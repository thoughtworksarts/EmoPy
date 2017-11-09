import os
from feature import Feature
import numpy as np
import csv
from skimage import io
from matplotlib import pyplot as plt
import cv2
from PIL import Image

class DataProcessor:
    """
    Class containing all necessary data preprocessing methods.
    """

    def get_image_features(self, from_csv, dataset_location, initial_image_dims, target_image_dims, feature_set, label_index=None, image_index=None, vector=True, time_series=True):

        orientations = 8
        pixels_per_cell = (16,16)
        cells_per_block=(1,1)
        hog_params = dict()
        hog_params['image_size'] = target_image_dims
        hog_params['orientations'] = orientations
        hog_params['pixels_per_cell'] = pixels_per_cell
        hog_params['cells_per_block'] = cells_per_block

        if from_csv:
            return self.get_image_feature_array_from_csv(dataset_location, feature_set, label_index, image_index, initial_image_dims, vector, hog_params=hog_params)
        else:
            if time_series:
                return self.get_time_series_image_feature_array_from_directory(dataset_location, feature_set, vector, hog_params=hog_params)
            else:
                return self.get_image_feature_array_from_directory(dataset_location, feature_set, vector, hog_params=hog_params)


    def get_image_feature_array_from_directory(self, root_directory, feature_set, vector=True, hog_params=None):
        """
        Extracts features vectors of all images found in root_directory.
        :param root_directory: location of image data
        :param vector: if true returns features as vectors, otherwise as 2D arrays
        :return: numpy array of extracted feature vectors
        """
        feature_type_index = 0 if vector else 1
        feature = Feature()
        features = list()
        for sub_directory in os.listdir(root_directory):
            if not sub_directory.startswith('.'):
                sub_directory_path = root_directory + '/' + sub_directory
                for image_file in os.listdir(sub_directory_path):
                    if not image_file.startswith('.'):
                        image_file_path = sub_directory_path + '/' + image_file
                        features.append(feature.extract_hog_feature(image_file=image_file_path, hog_params=hog_params)[feature_type_index])

        return np.array(features)

    def get_time_series_image_feature_array_from_directory(self, root_directory, feature_set, vector=True, hog_params=None):
        """
        Extracts features vectors of images found in root_directory and groups them
        by time_series batch. Subdirectories of root_directory must contain a single
        time series batch.
        :param root_directory: location of image data
        :param vector: if true returns features as vectors, otherwise as 2D arrays
        :return: numpy array of arrays which contain time series batch features
        """
        feature_type_index = 0 if vector else 1
        feature = Feature()
        features = list()
        for sub_directory in os.listdir(root_directory):
            if not sub_directory.startswith('.'):
                sub_directory_path = root_directory + '/' + sub_directory
                feature_batch = list()
                for image_file in os.listdir(sub_directory_path):
                    if not image_file.startswith('.'):
                        image_file_path = sub_directory_path + '/' + image_file
                        feature_batch.append(feature.extract_features(feature_type_index=feature_type_index, image_file=image_file_path, hog_params=hog_params))

                features.append(feature_batch)

        return np.array(features)

    def get_image_feature_array_from_csv(self, csv_file_path, feature_set,  label_index=0, image_index=1, image_dims=(48, 48), target_image_dims=(64,64), vector=True, hog_params=None):
        """
        Extracts features vectors of all images found in specified csv file.
        :param csv_file_path: location of dataset csv file
        :param label_index: column index of label value
        :param image_index: column index of image values
        :param image_dims: dimensions of image
        :param vector: if true returns features as vectors, otherwise as 2D arrays
        :return: numpy array of extracted feature vectors
        """
        feature_type_index = 0 if vector else 1

        feature = Feature()
        features = list()
        with open(csv_file_path) as csv_file:
            reader = csv.reader(csv_file, delimiter=',', quotechar='"')

            tempCount = 0

            for row in reader:
                if row[label_index] == 'emotion': continue
                image = np.asarray([int(pixel) for pixel in row[image_index].split(' ')], dtype=np.uint8).reshape(image_dims)
                image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_LINEAR)
                # io.imshow(image)
                # plt.show()

                # features.append(feature.extract_hog_feature(image_array=image)[feature_type_index])
                features.append(feature.extract_features(feature_type_index=feature_type_index, image_array=image, hog_params=hog_params))

                # break
                if tempCount == 4:  break
                tempCount += 1

        return np.array(features)
