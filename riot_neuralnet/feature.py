from skimage import color, io
from skimage.feature import hog


class Feature:

    def extract_hog_feature_vector(self, imageFile):
        image = io.imread(imageFile)
        image.resize((64,64))
        image = color.rgb2gray(image)
        featureVector, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualise=True) #, transform_sqrt=True, feature_vector=False)
        return featureVector, hog_image