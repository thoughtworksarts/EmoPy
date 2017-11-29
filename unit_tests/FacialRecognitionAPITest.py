import unittest, sys
import numpy as np
sys.path.append('../RIOTRuntime')
sys.path.append('../inception_v3_base')
from FacialRecognitionAPI import FacialRecognitionAPI
from keras.models import Model
from keras.models import load_model
from skimage import io, color
from EmotionEnum import Emotion
from keras.models import model_from_json


class FacialRecognitionAPITest(unittest.TestCase):

    def test_load_facial_recognition_api(self):
        # Setup
        with open('../trained_models/inception_v3_model_dl4j.json', 'r') as myfile:
            json_string=myfile.read().replace('\n', '')

        model = model_from_json(json_string)
        model.load_weights('../trained_models/inception_v3_weights_dl4j.h5')

        #model = load_model('../trained_models/inception_v3_model_1.h5')
        image_file = '../RIOTRuntime/images/testimage.png'
        image = io.imread(image_file)
        print ("shape is ")
        print (image.shape)
        image_to_test = np.expand_dims(image, axis=0)

        facial_recognition_api = FacialRecognitionAPI(image_to_test, model)

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