from FacialRecognitionUtil import FacialRecognitionUtil

class FacialRecognitionAPI(object):

    def __init__(self, input_source, model):
        self.input_source = input_source
        #self.input_source.load()
        self.trained_model = model
        self.utils = FacialRecognitionUtil()

    def get_prediction(self):
        prediction_values = self.trained_model.predict(self.input_source)
        print (type(prediction_values))
        return self.utils.translate(prediction_values[0])

