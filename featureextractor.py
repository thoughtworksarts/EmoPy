from skimage import color, io
from skimage.feature import hog, local_binary_pattern
import numpy as np


class FeatureExtractor:
    """

    :param images: list of images
    :param return_2d_array: if true, process features as 2d arrays, otherwise processes a 1d vectors
    """

    def __init__(self, images, return_2d_array=True):
        self.images = images
        self.return_2d_array = return_2d_array
        self.feature_params = dict()
        self.possible_features = ['hog', 'lbp']
        self.required_feature_parameters = dict()
        self.required_feature_parameters['hog'] = ['orientations', 'pixels_per_cell', 'cells_per_block']
        self.required_feature_parameters['lbp'] = ['radius', 'n_points']

    def add_feature(self, feature_type, params):
        """
        Adds the specified feature type and corresponding feature parameters to the list of types to be extracted.

        :param feature_type: feature type (e.g. 'hog')
        :param params: parameters corresponding to specified feature type

        ==================      =======================================================
            Possible Feature Types
        -------------------------------------------------------------------------------
            feature_type             required parameters
        ==================      =======================================================
            hog                      orientations, pixels_per_cell, cells_per_block
            lbp                      radius, n_points
        ==================      =======================================================
        """
        if feature_type not in self.possible_features:
            raise ValueError('Cannot extract specified feature. Use one of: ' + ', '.join(self.possible_features))
        if set(params.keys()) != set(self.required_feature_parameters[feature_type]):
            raise ValueError(('Expected %s parameters: ' + ', '.join(self.required_feature_parameters[feature_type])) % feature_type)
        self.feature_params[feature_type] = params

    def extract(self):
        """
        Extracts specified feature types from input training images.

        :return: list of features
        """
        features = list()
        for image in self.images:
            feature = list()
            for feature_type in self.feature_params.keys():
                feature += list(getattr(self, 'extract_%s_feature' % feature_type)(self.feature_params[feature_type], image)[self.return_2d_array])
            features.append(feature)
        return np.array(features)

    def extract_hog_feature(self, params, image):
        """
        Extracts Histogram of Oriented Gradients (HOG) feature from images based on specified parameters.

        :param params: user-supplied HOG feature parameters
        :param image: target image
        :return: feature vector/array
        """
        feature_vector, hog_image = hog(image, orientations=params['orientations'], pixels_per_cell=params['pixels_per_cell'], cells_per_block=params['cells_per_block'], visualise=True)
        return feature_vector, hog_image

    def extract_lbp_feature(self, params, image):
        """
        Extracts Local Binary Pattern (LBP) feature from images based on specified parameters.

        :param params: user-supplied LBP feature parameters
        :param image: target image
        :return: feature vector/array
        """
        feature_image = local_binary_pattern(image, params['n_points'], params['radius'])
        return feature_image.flatten(), feature_image
