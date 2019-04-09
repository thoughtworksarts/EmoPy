from keras.models import load_model
import cv2
from scipy import misc
import numpy as np
from pkg_resources import resource_filename

from EmoPy.src.face_detection import FaceDetector


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

    def __init__(self, target_emotions, model_file, verbose=False, face_detector=FaceDetector()):
        self.target_emotions = target_emotions
        self.emotion_index_map = {
            'anger': 0,
            'disgust': 1,
            'fear': 2,
            'happiness': 3,
            'sadness': 4,
            'surprise': 5,
            'calm': 6
        }
        self._check_emotion_set_is_supported()
        self.verbose = verbose
        self.target_dimensions = (48, 48)
        self.channels = 1
        self.face_detector = face_detector
        self.model = load_model(resource_filename('EmoPy', model_file))
        # self.model, self.emotion_map = self._choose_model_from_target_emotions()

    def predict(self, image_file):
        """
        Predicts discrete emotion for given image.

        :param images: image file (jpg or png format)
        """
        image = misc.imread(image_file)
        self.predict_from_ndarray(image)

    def predict_from_ndarray(self, image_array):
        """
        Predicts discrete emotion for given image.

        :param image_array: a n dimensional array representing an image
        """
        gray_image = image_array
        if len(image_array.shape) > 2:
            gray_image = cv2.cvtColor(image_array, code=cv2.COLOR_BGR2GRAY)
        cropped_image = self.face_detector.crop_face(gray_image)
        resized_image = cv2.resize(cropped_image, self.target_dimensions, interpolation=cv2.INTER_LINEAR)
        final_image = np.array([np.array([resized_image]).reshape(list(self.target_dimensions)+[self.channels])])
        prediction = self.model.predict(final_image)
        # Return the dominant expression
        dominant_expression = self._print_prediction(prediction[0])
        return dominant_expression

    def _check_emotion_set_is_supported(self):
        """
        Validates set of user-supplied target emotions.
        """
        supported_emotion_subsets = [
            {'anger', 'fear', 'surprise', 'calm'},
            {'happiness', 'disgust', 'surprise'},
            {'anger', 'fear', 'surprise'},
            {'anger', 'fear', 'calm'},
            {'anger', 'happiness', 'calm'},
            {'anger', 'fear', 'disgust'},
            {'calm', 'disgust', 'surprise'},
            {'sadness', 'disgust', 'surprise'},
            {'anger', 'happiness'}
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

    # def _choose_model_from_target_emotions(self):
    #     """
    #     Initializes pre-trained deep learning model for the set of target emotions supplied by user.
    #     """
    #     model_indices = [self.emotion_index_map[emotion] for emotion in self.target_emotions]
    #     sorted_indices = [str(idx) for idx in sorted(model_indices)]
    #     model_suffix = ''.join(sorted_indices)
    #     #Modify the path to choose the model file and the emotion map that you want to use
    #     model_file = 'models/conv_model_%s.hdf5' % model_suffix
    #     emotion_map_file = 'models/conv_emotion_map_%s.json' % model_suffix
    #     emotion_map = json.loads(open(resource_filename('EmoPy', emotion_map_file)).read())
    #     return load_model(resource_filename('EmoPy', model_file)), emotion_map

    def _print_prediction(self, prediction):
        normalized_prediction = [x/sum(prediction) for x in prediction]
        emotion_map = {emotion: index for emotion, index
                       in zip(sorted(self.target_emotions), range(0, len(self.target_emotions)))}
        for emotion in emotion_map.keys():
            print('%s: %.1f%%' % (emotion, normalized_prediction[emotion_map[emotion]]*100))
        dominant_emotion_index = np.argmax(prediction)
        dominant_emotion = ''
        for emotion in emotion_map.keys():
            if dominant_emotion_index == emotion_map[emotion]:
                dominant_emotion = emotion
                break
        return dominant_emotion
