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
        self.required_hog_parameters = ['orientations', 'pixels_per_cell', 'cells_per_block']
        self.required_lbp_parameters = ['radius', 'n_points']

    def add_feature(self, feature_type, params):
        if feature_type not in self.possible_features:
            raise ValueError('Cannot extract specified feature. Use one of: ' + ', '.join(self.possible_features))

        if feature_type is 'hog':
            if set(params.keys()) != set(self.required_hog_parameters):
                raise ValueError('Expected hog parameters: ' + ', '.join(self.required_hog_parameters))

        if feature_type is 'lbp':
            if set(params.keys()) != set(self.required_lbp_parameters):
                raise ValueError('Expected lbp parameters: ' + ', '.join(self.required_lbp_parameters))

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
