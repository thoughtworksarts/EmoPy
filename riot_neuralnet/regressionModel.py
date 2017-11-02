from data import get_image_feature_vector_array, get_training_label_array
from keras.layers import Dense
from keras.models import Sequential
import math

class RegressionModel:
    def __init__(self, features, labels):  # , num_samples, num_features):
        self.features = features
        self.labels = labels
        self.testFeatures = features[int(math.ceil(len(features)*0.75)):len(features)]
        self.testLabels = labels[int(math.ceil(len(labels)*0.75)):len(labels)]
        self.model = Sequential()
        self.model.add(Dense(4, input_shape=(128,), activation='sigmoid'))

    def fit(self):
        self.model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])

        self.model.fit(self.features, self.labels,
                  batch_size=1, epochs=2,
                  validation_split=0.25)

    def predict(self):
        return self.model.predict(self.testFeatures, batch_size=1)

def main():
    features = get_image_feature_vector_array()
    labels = get_training_label_array()
    model = RegressionModel(features, labels)
    model.fit()
    prediction = model.predict()

main()
