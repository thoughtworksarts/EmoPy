from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.layers import Dense, GlobalAveragePooling2D, Flatten, Conv2D, ConvLSTM2D
from keras.models import Model, Sequential


class FERNeuralNet(object):

    def __init__(self):
        self.model = self.init_model()

    def _init_model(self):
        raise NotImplementedError("Class %s doesn't implement init_model()" % self.__class__.__name__)

    def _fit(self, X_train, y_train, X_test, y_test):
        raise NotImplementedError("Class %s doesn't implement fit()" % self.__class__.__name__)

    def _predict(self):
        raise NotImplementedError("Class %s doesn't implement predict()" % self.__class__.__name__)


class TransferLearningNN(FERNeuralNet):

    def __init__(self, model_name='inception_v3'):
        self.model_name = model_name
        super().__init__()

    def init_model(self):

        # create the base pre-trained model
        base_model = None
        if self.model_name == 'inception_v3':
            base_model = InceptionV3(weights='imagenet', include_top=False)

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        # add a logistic layer -- FER+ has 7 prediction classes
        # (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral)
        predictions = Dense(units=7, activation='softmax')(x)

        # this is the model we will train
        model = Model(inputs=base_model.input, outputs=predictions)

        print(model.summary())

        # first: train only the top layers (which were randomly initialized)
        # i.e. freeze all convolutional InceptionV3 layers
        for layer in base_model.layers:
            layer.trainable = False

        # compile the model (should be done *after* setting layers to non-trainable)
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

        return model

    def fit(self, X_train, y_train, X_test, y_test):
        self.model.fit(x=X_train, y=y_train, epochs=50, verbose=1, callbacks=[ReduceLROnPlateau(), EarlyStopping(patience=3)], validation_data=(X_test, y_test), shuffle=True)

        for layer in self.model.layers[:249]:
            layer.trainable = False
        for layer in self.model.layers[249:]:
            layer.trainable = True

        self.model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(x=X_train, y=y_train, epochs=50, verbose=1, callbacks=[ReduceLROnPlateau(), EarlyStopping(patience=3)], validation_data=(X_test, y_test), shuffle=True)


class TimeDelayNN(FERNeuralNet):

    def __init__(self, time_delay=3, num_output_values=4, verbose=False):
        self.time_delay = time_delay
        self.num_output_values = num_output_values
        self.verbose = verbose
        super().__init__()

    def init_model(self):
        model = Sequential()
        model.add(Conv2D(filters=10, kernel_size=(self.time_delay, self.num_output_values), activation="sigmoid", input_shape=(1,self.time_delay,self.num_output_values), padding="same"))
        model.add(Flatten())
        model.add(Dense(units=4, activation="sigmoid"))
        if self.verbose:
            model.summary()
        return model

    def fit(self, X_train, y_train, X_test, y_test):
        self.model.compile(optimizer="RMSProp", loss="cosine_proximity",
                         metrics=["accuracy"])
        self.model.fit(X_train, y_train, batch_size=10, epochs=100, validation_data=(X_test, y_test),
                     callbacks=[ReduceLROnPlateau(), EarlyStopping(patience=3)])


class ConvolutionalLstmNN(FERNeuralNet):

    def __init__(self, image_size, channels, time_delay=2, verbose=False):
        self.time_delay = time_delay
        self.channels = channels
        self.image_size = image_size
        self.verbose = verbose
        super().__init__()

    def init_model(self):
        model = Sequential()
        model.add(ConvLSTM2D(filters=10, kernel_size=(4, 4), activation="sigmoid", input_shape=(self.time_delay, self.channels)+self.image_size, data_format='channels_first'))
        model.add(Flatten())
        model.add(Dense(units=4, activation="sigmoid"))
        if self.verbose:
            model.summary()
        return model

    def fit(self, X_train, y_train, X_test, y_test):
        self.model.compile(optimizer="RMSProp", loss="cosine_proximity", metrics=["accuracy"])
        self.model.fit(X_train, y_train, batch_size=10, epochs=100, validation_data=(X_test, y_test),
            callbacks=[ReduceLROnPlateau(), EarlyStopping(patience=3)])
