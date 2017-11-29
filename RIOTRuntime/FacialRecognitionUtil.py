import operator

class FacialRecognitionUtil(object):

    def translate(self, prediction_values):
        max_emotion = None
        max_value = float('-inf')
        print(prediction_values)
        for emotion, value in enumerate(prediction_values):
            if value > max_value:
                max_emotion = emotion
                max_value = value
        print (max_emotion)
        print (max_value)
        return max_emotion

        # return max(prediction_values.iteritems(), key=operator.itemgetter(1))[0]