from skimage import color, io
from skimage.feature import hog, local_binary_pattern
import numpy as np


class Feature:

    def extract_features(self, target_image_dims, feature_params, feature_type_index=0, image_file=None, image_array=None, image_size=None):
        features = list()
        for feature_type in feature_params.keys():
            feature = list(getattr(self, 'extract_%s_feature' % feature_type)(feature_params[feature_type], target_image_dims, image_file, image_array)[feature_type_index])
            features += feature
        return features

    def extract_hog_feature(self, hog_params, target_image_dims, image_file=None, image_array=None):
        image = image_array
        if image_file:
            image = io.imread(image_file)
            image.resize(target_image_dims)
            image = color.rgb2gray(image)
        feature_vector, hog_image = hog(image, orientations=hog_params['orientations'], pixels_per_cell=hog_params['pixels_per_cell'], cells_per_block=hog_params['cells_per_block'], visualise=True)

        return feature_vector, hog_image

    def extract_lbp_feature(self, lbp_params, target_image_dims, image_file=None, image_array=None):
        image = image_array
        if image_file:
            image = io.imread(image_file)
            image.resize(target_image_dims)
            image = color.rgb2gray(image)
        feature_image = local_binary_pattern(image, lbp_params['n_points'], lbp_params['radius'])

        return feature_image.flatten(), feature_image
