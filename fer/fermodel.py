

class FERModel:
    POSSIBLE_EMOTIONS = ['anger', 'fear', 'calm', 'sad', 'happy', 'surprise']

    def __init__(self, target_emotions):
        if not self._emotions_are_valid(target_emotions):
            raise ValueError('Target emotions must be subset of %s.' % self.POSSIBLE_EMOTIONS)
        self.target_emotions = target_emotions
        self._initialize_model()

    def _initialize_model(self):
        print('Initializing FER model for target emotions: %s' % self.target_emotions)

    def train(self):
        pass

    def predict(self, images):
        pass

    def _emotions_are_valid(self, emotions):
        return set(emotions).issubset(set(self.POSSIBLE_EMOTIONS))