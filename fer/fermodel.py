from neuralnets import *
from imageprocessor import ImageProcessor
from featureextractor import FeatureExtractor
import numpy as np


class FERModel:
    """
    Deep learning model for facial expression recognition. Model chosen dependent on set of target emotions supplied by user.

    :param target_emotions: set of target emotions to classify
    :param train_images: numpy array of training images
    :param csv_data_path: local path to directory containing csv with image pixel values
    :param verbose: if true, will print out extra process information
    """

    POSSIBLE_EMOTIONS = ['anger', 'fear', 'calm', 'sad', 'happy', 'surprise']

    def __init__(self, target_emotions, train_images=None, train_labels=None, csv_data_path=None, verbose=False):
        if not self._emotions_are_valid(target_emotions):
            raise ValueError('Target emotions must be subset of %s.' % self.POSSIBLE_EMOTIONS)
        if not train_images and not csv_data_path:
            raise ValueError('Must supply training images or datapath containing training images.')
        self.target_emotions = target_emotions
        self.train_images = train_images
        self.x_train = train_images
        self.y_train = train_labels
        self.verbose = verbose
        self.time_delay = 1
        self.raw_dimensions = (48, 48)
        self.target_dimensions = (64, 64)
        self.channels = 1

        if csv_data_path:
            self._extract_training_images_from_path(csv_data_path)

        self._initialize_model()

    def _initialize_model(self):
        print('Initializing FER model parameters for target emotions: %s' % self.target_emotions)
        self.model = self._choose_model_from_target_emotions()

    def train(self):
        """
        Trains FERModel on supplied image data.
        """
        print('Training FERModel...')
        validation_split = 0.15
        self.model.fit(self.x_train, self.y_train, validation_split)

    def predict(self, images):
        pass

    def _emotions_are_valid(self, emotions):
        """
        Validates set of user-supplied target emotions
        :param emotions: list of emotions supplied by user
        :return: true if emotion set is valid, false otherwise
        """
        return set(emotions).issubset(set(self.POSSIBLE_EMOTIONS))

    def _extract_training_images_from_path(self, csv_data_path):
        """
        Extracts training images from csv file found in user-supplied directory path
        :param csv_data_path: path to directory containing image data csv file supplied by user
        """
        print('Extracting training images from path...')
        imageProcessor = ImageProcessor(from_csv=True, datapath=csv_data_path, target_dimensions=self.target_dimensions, raw_dimensions=self.raw_dimensions, csv_label_col=0, csv_image_col=1, channels=1)
        images, labels = imageProcessor.get_training_data()
        self.train_images = images
        self.y_train = labels

        if self.verbose:
            print(images.shape)
            print(labels.shape)

    def _choose_model_from_target_emotions(self):
        """
        Chooses best-performing deep learning model for the set of target emotions supplied by user.
        :return: One of deep learning models from neuralnets.py
        """
        print('Creating FER model...')
        # TODO: call _extract_features for appropriate models
        if self.target_emotions == self.POSSIBLE_EMOTIONS:
            self._extract_features()
            return ConvolutionalLstmNN(self.target_dimensions, self.channels, time_delay=self.time_delay)
        # TODO: add conditionals to choose best models for all emotion subsets

    def _extract_features(self):
        """
        Extract best-performing features from images for model. If called, features will be used for training rather than the raw images.
        """
        print('Extracting features from training images...')
        temp_images = list()
        for image in self.train_images:
            temp_images.append(image[0])
        images = temp_images

        featureExtractor = FeatureExtractor(images, return_2d_array=True)
        featureExtractor.add_feature('hog', {'orientations': 8, 'pixels_per_cell': (16, 16), 'cells_per_block': (1, 1)})
        #featureExtractor.add_feature('lbp', {'n_points': 24, 'radius': 3})
        raw_features = featureExtractor.extract()
        features = list()
        for feature in raw_features:
            features.append([[feature]])
        self.x_train = np.array(features)
        if self.verbose:
            print('feature shape: ' + str(self.x_train.shape))
            print('label shape: ' + str(self.y_train.shape))
