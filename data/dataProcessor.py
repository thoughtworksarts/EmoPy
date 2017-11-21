import os, csv, cv2, math, datetime

import numpy as np
from skimage import io
from matplotlib import pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

from feature import Feature



class DataProcessor:
    """
    Class containing all necessary data preprocessing methods.
    """

    def __init__(self):
        self.feature_parameters = dict()
        self.possible_features = ['hog', 'lbp']
        self.required_feature_parameters = dict()
        self.required_feature_parameters['hog'] = ['orientations', 'pixels_per_cell', 'cells_per_block']
        self.required_feature_parameters['lbp'] = ['radius', 'n_points']

    def add_feature(self, feature_type, params):
        if feature_type not in self.possible_features:
            raise ValueError('Cannot extract specified feature. Use one of: ' + ', '.join(self.possible_features))

        if set(params.keys()) != set(self.required_feature_parameters[feature_type]):
            raise ValueError(('Expected %s parameters: ' + ', '.join(self.required_feature_parameters[feature_type])) % feature_type)

        # if feature_type is 'lbp':
        #     if set(params.keys()) != set(self.required_lbp_parameters):
        #         raise ValueError('Expected lbp parameters: ' + ', '.join(self.required_lbp_parameters))

        self.feature_parameters[feature_type] = params

    def get_training_data(self, from_csv, dataset_location, target_image_dims, initial_image_dims=None, label_index=None, image_index=None, vector=True, time_series=True, test_data_percentage=0.20):

        if from_csv:
            return self.get_training_data_from_csv(dataset_location, initial_image_dims, target_image_dims, label_index, image_index, vector, test_data_percentage)
        else:
            if time_series:
                return self.get_time_series_image_feature_array_from_directory(dataset_location, target_image_dims, vector)
            else:
                return self.get_image_feature_array_from_directory(dataset_location, target_image_dims, vector)


    def get_image_feature_array_from_directory(self, root_directory, target_image_dims, vector=True):
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
                        features.append(feature.extract_features(target_image_dims, self.feature_parameters, feature_type_index=feature_type_index, image_file=image_file_path))


        return np.array(features)

    def get_time_series_image_feature_array_from_directory(self, root_directory, target_image_dims, vector=True):
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
                        feature_batch.append(feature.extract_features(target_image_dims, self.feature_parameters, feature_type_index=feature_type_index, image_file=image_file_path))


                features.append(feature_batch)

        return np.array(features)

    def get_training_data_from_csv(self, csv_file_path, image_dims, target_image_dims, label_index=0, image_index=1, vector=True, test_data_percentage=0.20):
        """
        Extracts features vectors of all images found in specified csv file.
        :param csv_file_path: location of dataset csv file
        :param image_dims: dimensions of image
        :param label_index: column index of label value
        :param image_index: column index of image values
        :param vector: if true returns features as vectors, otherwise as 2D arrays
        :return: numpy array of extracted feature vectors
        """

        print('Extracting training data from csv...')
        start = datetime.datetime.now()

        feature_type_index = 0 if vector else 1

        feature = Feature()
        features = list()
        labels = list()
        with open(csv_file_path) as csv_file:
            reader = csv.reader(csv_file, delimiter=',', quotechar='"')

            tempCount = 0

            for row in reader:
                if row[label_index] == 'emotion': continue

                label = [0]*7
                label[int(row[label_index])] = 1.0
                labels.append(np.array(label))

                image = np.asarray([int(pixel) for pixel in row[image_index].split(' ')], dtype=np.uint8).reshape(image_dims)
                image = cv2.resize(image, target_image_dims, interpolation=cv2.INTER_LINEAR)
                image_3d = np.array([image, image, image]).reshape((target_image_dims[0], target_image_dims[1], 3))

                # image = np.array(feature.extract_features(target_image_dims, self.feature_parameters, feature_type_index=feature_type_index, image_array=image))

                # io.imshow(image)
                # plt.show()

                # image_3d = np.array([image, image, image]).reshape((target_image_dims[0], target_image_dims[1], 3))
                features.append(image_3d)

                if tempCount == 9:  break   # for now only processing 10 images, o/w training will take too long
                tempCount += 1


        X_test = np.array(features[int(math.ceil(len(features)*(1-test_data_percentage))):len(features)])
        X_train = np.array(features[0:int(math.ceil(len(features)*(1-test_data_percentage)))])
        y_test = np.array(labels[int(math.ceil(len(labels)*(1-test_data_percentage))):len(labels)])
        y_train = np.array(labels[0:int(math.ceil(len(labels)*(1-test_data_percentage)))])

        data_gen = ImageDataGenerator(rotation_range=180)

        data_gen.fit(X_train)

        end = datetime.datetime.now()
        print('Training data extraction runtime - ' + str(end-start))

        return X_train, y_train, X_test, y_test

    def get_training_label_array(self):
        raw_training_labels = self.get_raw_training_labels()
        training_label_array = list()
        for time_series_key in raw_training_labels:
            time_series = raw_training_labels[time_series_key]
            training_label_array += time_series

        return np.array(training_label_array)

    def get_raw_training_labels(self):
        # Uses 20 photo series from the Cohn-Kanade dataset
        # hand labeled by AP
        # arousal(least, most), valence(negative, positive), power, anticipation
        raw_training_labels = {1: [10, [.6, .4, .7, .6], [.9, .1, .8, .9]],
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
        for time_series_key in raw_training_labels:
            time_series = raw_training_labels[time_series_key]
            num_images = time_series[0]
            increment = [(time_series[2][emotion_dimension_idx] - time_series[1][emotion_dimension_idx]) / num_images for emotion_dimension_idx in range(EMOTION_DIMENSION_COUNT)]
            labels[time_series_key] = [[increment[label_idx]*image_idx + (time_series[1][label_idx]) for label_idx in range(EMOTION_DIMENSION_COUNT)] for image_idx in range(num_images)]

        return labels