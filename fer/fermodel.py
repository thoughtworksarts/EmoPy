from neuralnets import *
from imageprocessor import ImageProcessor
from featureextractor import FeatureExtractor
import numpy as np


class FERModel:

    POSSIBLE_EMOTIONS = ['anger', 'fear', 'calm', 'sad', 'happy', 'surprise']

    def __init__(self, target_emotions, train_images=None, data_path=None, extract_features=False, verbose=False):
        if not self._emotions_are_valid(target_emotions):
            raise ValueError('Target emotions must be subset of %s.' % self.POSSIBLE_EMOTIONS)
        if not train_images and not data_path:
            raise ValueError('Must supply training images or datapath containing training images.')
        self.target_emotions = target_emotions
        self.train_images = train_images
        self.extract_features = extract_features
        self.verbose = verbose
        self.time_delay = 1
        self.raw_dimensions = (48, 48)
        self.target_dimensions = (64, 64)
        self.channels = 1

        if data_path:
            self._extract_training_images_from_path(data_path)

        self._initialize_model()

    def _initialize_model(self):
        print('Initializing FER model parameters for target emotions: %s' % self.target_emotions)
        self.model = self._choose_model_from_target_emotions()

    def train(self):
        if self.extract_features:
            self._extract_features()
        else:
            self.x_train = self.train_images
        print('Training FERModel...')
        validation_split = 0.15
        self.model.fit(self.x_train, self.y_train, validation_split)

    def predict(self, images):
        pass

    def _emotions_are_valid(self, emotions):
        return set(emotions).issubset(set(self.POSSIBLE_EMOTIONS))

    def _extract_training_images_from_path(self, data_path):
        print('Extracting training images from path...')
        imageProcessor = ImageProcessor(from_csv=True, datapath=data_path, target_dimensions=self.target_dimensions, raw_dimensions=self.raw_dimensions, csv_label_col=0, csv_image_col=1, channels=1)
        images, labels = imageProcessor.get_training_data()
        self.train_images = images
        self.y_train = labels

        if self.verbose:
            print(images.shape)
            print(labels.shape)

    def _choose_model_from_target_emotions(self):
        print('Creating FER model...')
        if self.target_emotions == self.POSSIBLE_EMOTIONS:
            return ConvolutionalLstmNN(self.target_dimensions, self.channels, time_delay=self.time_delay)
        # TODO: add conditionals to choose best models for all emotion subsets

    def _extract_features(self):
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
