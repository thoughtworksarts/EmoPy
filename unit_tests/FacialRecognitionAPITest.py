import unittest, sys
sys.path.append('../RIOTRuntime')
sys.path.append('../inception_v3_base')
from FacialRecognitionAPI import FacialRecognitionAPI
from keras.models import Model
from keras.models import load_model
from skimage import io, color
from EmotionEnum import Emotion

class FacialRecognitionAPITest(unittest.TestCase):

    def test_load_facial_recognition_api(self):
        # Setup
        model = load_model('../trained_models/inception_v3_model_1.h5')

        image_file = '../RIOTRuntime/images/Greyscale.jpg'
        image = io.imread(image_file)

        facial_recognition_api = FacialRecognitionAPI(image, model)

        # Act
        prediction = facial_recognition_api.get_prediction()

        # Assert
        self.assertEqual(type(prediction), Emotion)




    # def test_isupper(self):
    #     self.assertTrue('FOO'.isupper())
    #     self.assertFalse('Foo'.isupper())
    #
    # def test_split(self):
    #     s = 'hello world'
    #     self.assertEqual(s.split(), ['hello', 'world'])
    #     # check that s.split fails when the separator is not a string
    #     with self.assertRaises(TypeError):
    #         s.split(2)

if __name__ == '__main__':
    unittest.main()