from data import get_image_feature_vector_array, get_training_label_array
from keras.layers import Dense
from keras.models import Sequential

class RegressionModel:
    def __init__(self, features, labels):  # , num_samples, num_features):
        self.features = features
        self.labels = labels
        self.model = Sequential()
        self.model.add(Dense(4, input_shape=(128,), activation='sigmoid'))

    def fit(self):
        self.model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])

        self.model.fit(self.features, self.labels,
                  batch_size=1, epochs=50,
                  validation_split=0.25)

def predict(self):
    print(model.predict(X_train_scale, batch_size=1))

def main():
    features = get_image_feature_vector_array()
    labels = get_training_label_array()
    model = RegressionModel(features, labels)
    model.fit()
    # prediction = model.predict()
    # print prediction

main()
