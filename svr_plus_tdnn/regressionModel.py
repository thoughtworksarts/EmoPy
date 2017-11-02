from keras.layers import Dense
from keras.models import Sequential
import math

class RegressionModel:
    def __init__(self, features, labels, num_output_values=4, test_data_percentage=0.25):
        self.test_data_percentage = test_data_percentage
        self.features = features
        self.labels = labels
        self.test_features = features[int(math.ceil(len(features) * (1 - test_data_percentage))):len(features)]
        self.test_labels = labels[int(math.ceil(len(labels) * (1 - test_data_percentage))):len(labels)]
        self.feature_vector_length = len(features[0])

        self.model = Sequential()
        self.model.add(Dense(num_output_values, input_shape=(self.feature_vector_length,), activation="sigmoid"))

    def fit(self):
        self.model.compile(loss="mean_squared_error", optimizer="rmsprop", metrics=["accuracy"])

        self.model.fit(self.features, self.labels,
                  batch_size=1, epochs=2,
                  validation_split=self.test_data_percentage)

    def predict(self):
        return self.model.predict(self.test_features, batch_size=1)
