from skimage import color, io
from skimage.feature import hog


class Feature:

    def extract_features(self, feature_type_index=0, image_file=None, image_array=None, hog_params=None, lbp_params=None):
        features = list()
        if hog_params: features.append(self.extract_hog_feature(hog_params, image_file, image_array)[feature_type_index])
        if lbp_params: features.append(self.extract_lbp_feature(lbp_params, image_file, image_array)[feature_type_index])
        return features

    def extract_hog_feature(self, hog_params, image_file=None, image_array=None):
        image = image_array
        if image_file:
            image = io.imread(image_file)
            image.resize(hog_params['image_size'])
            image = color.rgb2gray(image)
        featureVector, hog_image = hog(image, orientations=hog_params['orientations'], pixels_per_cell=hog_params['pixels_per_cell'], cells_per_block=hog_params['cells_per_block'], visualise=True)
        return featureVector, hog_image

    def extract_lbp_feature(self, feature_type_index=0, image_file=None, image_array=None):
        pass
