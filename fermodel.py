from src.neuralnets import *

from keras.models import model_from_json, Model

class FERModel:
    """
    Pretrained deep learning model for facial expression recognition.

    :param target_emotions: set of target emotions to classify
    :param verbose: if true, will print out extra process information

    **Example**::

        from fermodel import FERModel

        target_emotions = ['happiness', 'disgust', 'surprise']
        model = FERModel(target_emotions, verbose=True)

    """

    POSSIBLE_EMOTIONS = ['anger', 'fear', 'calm', 'sadness', 'happiness', 'surprise', 'disgust']

    def __init__(self, target_emotions, verbose=False):
        self.target_emotions = target_emotions
        self.emotion_index_map = {
            'anger': 0,
            'disgust': 1,
            'fear': 2,
            'happiness': 3,
            'sadness': 4,
            'surprise': 5,
            'neutral': 6
        }
        self._check_emotion_set_is_supported()
        self.verbose = verbose
        self.target_dimensions = (48, 48)
        self._initialize_model()

    def _initialize_model(self):
        print('Initializing FER model parameters for target emotions: %s' % self.target_emotions)
        self.model = self._choose_model_from_target_emotions()

    def predict(self, images):
        """
        Predicts discrete emotions for given images.

        :param images: list of images
        """
        pass

    def _check_emotion_set_is_supported(self):
        """
        Validates set of user-supplied target emotions.
        """
        supported_emotion_subsets = [
            set(['anger', 'fear', 'surprise', 'calm']),
            set(['anger', 'fear', 'calm']),
            set(['anger', 'happiness', 'calm']),
            set(['anger', 'fear', 'surprise']),
            set(['anger', 'fear', 'disgust']),
            set(['anger', 'fear', 'sadness']),
            set(['happiness', 'disgust', 'surprise']),
            set(['anger', 'surprise']),
            set(['fear', 'surprise']),
            set(['calm', 'disgust', 'surprise']),
            set(['sadness', 'disgust', 'surprise']),
            set(['anger', 'disgust']),
            set(['anger', 'fear']),
            set(['disgust', 'surprise'])
        ]
        if not set(self.target_emotions) in supported_emotion_subsets:
            error_string = 'Target emotions must be a supported subset. '
            error_string += 'Choose from one of the following emotion subset: \n'
            possible_subset_string = ''
            for emotion_set in supported_emotion_subsets:
                possible_subset_string += ', '.join(emotion_set)
                possible_subset_string += '\n'
            error_string += possible_subset_string
            raise ValueError(error_string)

    def _choose_model_from_target_emotions(self):
        """
        Initializes pre-trained deep learning model for the set of target emotions supplied by user.
        """
        print('Initializing FER model...')

        model_indices = [self.emotion_index_map[emotion] for emotion in self.target_emotions]
        sorted_indices = [str(idx) for idx in sorted(model_indices)]
        model_suffix = ''.join(sorted_indices)

        model_file = open('../models/conv_model_%s.json' % model_suffix,'r')
        weights_file = '../models/conv_weights_%s.h5' % model_suffix

        self.model = model_from_json(model_file.read())
        self.model.load_weights(weights_file)