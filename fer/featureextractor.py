from skimage import color, io
from skimage.feature import hog, local_binary_pattern
import numpy as np


class FeatureExtractor:

    def __init__(self, images, return_2d_array=True):
        # user-supplied parameters
        self.images = images
        self.return_2d_array = return_2d_array
        self.feature_params = dict()

        # feature requirements
        self.possible_features = ['hog', 'lbp']
        self.required_feature_parameters = dict()
        self.required_feature_parameters['hog'] = ['orientations', 'pixels_per_cell', 'cells_per_block']
        self.required_feature_parameters['lbp'] = ['radius', 'n_points']

    def add_feature(self, feature_type, params):
        if feature_type not in self.possible_features:
            raise ValueError('Cannot extract specified feature. Use one of: ' + ', '.join(self.possible_features))
        if set(params.keys()) != set(self.required_feature_parameters[feature_type]):
            raise ValueError(('Expected %s parameters: ' + ', '.join(self.required_feature_parameters[feature_type])) % feature_type)
        self.feature_params[feature_type] = params

    def extract(self):
        features = list()
        for image in self.images:
            feature = list()
            for feature_type in self.feature_params.keys():
                feature += list(getattr(self, 'extract_%s_feature' % feature_type)(self.feature_params[feature_type], image)[self.return_2d_array])
            features.append(feature)
        return np.array(features)

    def _extract_hog_feature(self, params, image):
        feature_vector, hog_image = hog(image, orientations=params['orientations'], pixels_per_cell=params['pixels_per_cell'], cells_per_block=params['cells_per_block'], visualise=True)
        return feature_vector, hog_image

    def _extract_lbp_feature(self, params, image):
        feature_image = local_binary_pattern(image, params['n_points'], params['radius'])
        return feature_image.flatten(), feature_image
