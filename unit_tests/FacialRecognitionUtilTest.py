import unittest, sys
sys.path.append('../RIOTRuntime')
sys.path.append('../inception_v3_base')
from EmotionEnum import Emotion
from FacialRecognitionUtil import FacialRecognitionUtil


class FacialRecognitionUtilTest(unittest.TestCase):
    def test_translate_calm(self):
        # Setup
        facial_recognition_util = FacialRecognitionUtil()
        prediction_values = {Emotion.ANGRY: 0.12, Emotion.FEAR: 0.22, Emotion.CALM: 0.6}

        # Act
        result = facial_recognition_util.translate(prediction_values)

        # Assert
        self.assertEqual(result, Emotion.CALM)

if __name__ == '__main__':
    unittest.main()