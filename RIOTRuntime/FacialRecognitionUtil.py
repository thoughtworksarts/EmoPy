import operator

class FacialRecognitionUtil(object):

    def translate(self, prediction_values):
        return max(prediction_values.iteritems(), key=operator.itemgetter(1))[0]