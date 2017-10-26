from skimage import color, io
from skimage.feature import hog


class Feature:

    def extractFeatureVector(self, imageFile):
        image = io.imread(imageFile)
        image.resize((28,28))
        image = color.rgb2gray(image)
        featureVector, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualise=True) #, transform_sqrt=True, feature_vector=False)
        return hog_image