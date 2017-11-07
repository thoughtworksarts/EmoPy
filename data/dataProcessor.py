import os
from feature import Feature
import numpy as np


class DataProcessor:
    """
    Class containing all necessary data preprocessing methods.
    """

    def get_image_feature_array(self, root_directory, vector=True):
        """
        Extracts features vectors of all images found in root_directory.
        :param root_directory: location of image data
        :param vector: if true returns features as vectors, otherwise as 2D arrays
        :return: numpy array of extracted feature vectors
        """
        feature_index = 0 if vector else 1
        feature = Feature()
        features = list()
        for sub_directory in os.listdir(root_directory):
            if not sub_directory.startswith('.'):
                sub_directory_path = root_directory + '/' + sub_directory
                for image_file in os.listdir(sub_directory_path):
                    if not image_file.startswith('.'):
                        image_file_path = sub_directory_path + '/' + image_file
                        features.append(feature.extract_hog_feature_vector(image_file_path)[feature_index])

        return np.array(features)

    def get_time_series_image_feature_array(self, root_directory, vector=True):
        """
        Extracts features vectors of images found in root_directory and groups them
        by time_series batch. Subdirectories of root_directory must contain a single
        time series batch.
        :param root_directory: location of image data
        :param vector: if true returns features as vectors, otherwise as 2D arrays
        :return: numpy array of arrays which contain time series batch features
        """
        feature_index = 0 if vector else 1
        feature = Feature()
        features = list()
        for sub_directory in os.listdir(root_directory):
            if not sub_directory.startswith('.'):
                sub_directory_path = root_directory + '/' + sub_directory
                feature_batch = list()
                for image_file in os.listdir(sub_directory_path):
                    if not image_file.startswith('.'):
                        image_file_path = sub_directory_path + '/' + image_file
                        feature_batch.append(feature.extract_hog_feature_vector(image_file_path)[feature_index])
                features.append(feature_batch)

        return np.array(features)

