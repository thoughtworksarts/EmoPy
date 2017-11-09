from skimage import color, io
from skimage.feature import hog


class Feature:

    def extract_features(self, target_image_dims, feature_params, feature_type_index=0, image_file=None, image_array=None, image_size=None):
        features = list()
        if 'hog' in feature_params.keys(): features.append(self.extract_hog_feature(feature_params['hog'], target_image_dims, image_file, image_array)[feature_type_index])
        if 'lbp' in feature_params.keys(): features.append(self.extract_lbp_feature(feature_params['lbp'], image_file, image_array)[feature_type_index])
        return features

    def extract_hog_feature(self, hog_params, target_image_dims, image_file=None, image_array=None):
        image = image_array
        if image_file:
            image = io.imread(image_file)
            image.resize(target_image_dims)
            image = color.rgb2gray(image)
        featureVector, hog_image = hog(image, orientations=hog_params['orientations'], pixels_per_cell=hog_params['pixels_per_cell'], cells_per_block=hog_params['cells_per_block'], visualise=True)
        return featureVector, hog_image

    def extract_lbp_feature(self, feature_type_index=0, image_file=None, image_array=None):
        pass
