from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.layers import Dense, GlobalAveragePooling2D, Flatten, Conv2D, ConvLSTM2D, Conv3D
from keras.layers.normalization import BatchNormalization
from keras.models import Model, Sequential
from imageprocessor import ImageProcessor

class _FERNeuralNet(object):
    """
    Interface for all FER deep neural net classes.
    """

    def __init__(self):
        self.model = None
        self._init_model()

    def _init_model(self):
        raise NotImplementedError("Class %s doesn't implement _init_model()" % self.__class__.__name__)

    def fit(self, X_train, y_train, X_test, y_test):
        raise NotImplementedError("Class %s doesn't implement fit()" % self.__class__.__name__)

    def predict(self):
        raise NotImplementedError("Class %s doesn't implement predict()" % self.__class__.__name__)


class TransferLearningNN(_FERNeuralNet):
    """
    Convolutional Neural Network initialized with pretrained weights.

    :param image_size: dimensions of input images
    :param channels: number of image channels
    :param time_delay: number time steps for lookback
    :param model_name: name of pretrained model to use for initial weights. Options: ['Xception', 'VGG16', 'VGG19', 'ResNet50', 'InceptionV3', 'InceptionResNetV2']
    """
    #TODO: update constructor parameters
    def __init__(self, model_name, target_labels):
        self.model_name = model_name
        self.target_labels = target_labels
        super().__init__()

    def _init_model(self):

        # create the base pre-trained model
        base_model = None
        if self.model_name == 'inception_v3':
            base_model = InceptionV3(weights='imagenet', include_top=False)

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        # add a logistic layer -- FER+ has 7 prediction classes
        # (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral)
        predictions = Dense(units=len(self.target_labels), activation='softmax')(x)

        # this is the model we will train
        model = Model(inputs=base_model.input, outputs=predictions)

        print(model.summary())

        # first: train only the top layers (which were randomly initialized)
        # i.e. freeze all convolutional InceptionV3 layers
        for layer in base_model.layers:
            layer.trainable = False

        # compile the model (should be done *after* setting layers to non-trainable)
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

        self.model = model

    def fit(self, features, labels, validation_split):
        """
        Trains the neural net on the data provided.

        :param features: Numpy array of training data.
        :param labels: Numpy array of target (label) data.
        :param validation_split: Float between 0 and 1. Percentage of training data to use for validation
        """
        self.model.fit(x=features, y=labels, epochs=50, verbose=1, callbacks=[ReduceLROnPlateau(), EarlyStopping(patience=3)], validation_split=validation_split, shuffle=True)

        for layer in self.model.layers[:249]:
            layer.trainable = False
        for layer in self.model.layers[249:]:
            layer.trainable = True

        self.model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(x=features, y=labels, epochs=50, verbose=1, callbacks=[ReduceLROnPlateau(), EarlyStopping(patience=3)], validation_split=validation_split, shuffle=True)


class TimeDelayNN(_FERNeuralNet):

    def __init__(self, feature_vector_length, time_delay=3, num_output_values=4, verbose=False):
        self.time_delay = time_delay
        self.num_output_values = num_output_values
        self.feature_vector_length = feature_vector_length
        self.verbose = verbose
        self.regression_model = None
        super().__init__()

    def _init_model(self):
        self._init_regression_model()
        self._init_neural_net_model()

    def _init_regression_model(self):
        model = Sequential()
        model.add(Dense(self.num_output_values, input_shape=(self.feature_vector_length,), activation="sigmoid"))
        self.regression_model = model

    def _init_neural_net_model(self):

        model = Sequential()
        model.add(Conv2D(filters=10, kernel_size=(self.time_delay, self.num_output_values), activation="sigmoid", input_shape=(1,self.time_delay,self.num_output_values), padding="same"))
        model.add(Flatten())
        model.add(Dense(units=4, activation="sigmoid"))
        if self.verbose:
            model.summary()
        self.model = model


    def fit(self, features, labels, validation_split):
        """
        Trains the neural net on the data provided.

        :param features: Numpy array of training data.
        :param labels: Numpy array of target (label) data.
        :param validation_split: Float between 0 and 1. Percentage of training data to use for validation
        """
        self.regression_model.compile(loss="mean_squared_error", optimizer="rmsprop", metrics=["accuracy"])
        self.regression_model.fit(features, labels,
                       batch_size=1, epochs=20,
                       validation_split=0.0, shuffle=True)

        regression_predictions = self.regression_model.predict(features)
        imageProcessor = ImageProcessor()
        features, labels = imageProcessor.get_time_delay_training_data(regression_predictions, regression_predictions)
        self.model.compile(optimizer="RMSProp", loss="cosine_proximity", metrics=["accuracy"])
        self.model.fit(features, labels, batch_size=10, epochs=100, validation_split=validation_split, callbacks=[ReduceLROnPlateau(), EarlyStopping(patience=3)])


class ConvolutionalLstmNN(_FERNeuralNet):
    """
    Convolutional Long Short Term Memory Neural Network.

    :param image_size: dimensions of input images
    :param channels: number of image channels
    :param time_delay: number time steps for lookback
    """

    def __init__(self, image_size, channels, target_labels, time_delay=2, verbose=False):
        self.time_delay = time_delay
        self.channels = channels
        self.image_size = image_size
        self.target_labels = target_labels
        self.verbose = verbose
        super().__init__()

    def _init_model(self):
        model = Sequential()
        model.add(ConvLSTM2D(filters=10, kernel_size=(4, 4), activation="sigmoid", input_shape=(self.time_delay, self.channels)+self.image_size, data_format='channels_first', return_sequences=True))
        model.add(BatchNormalization())
        model.add(ConvLSTM2D(filters=10, kernel_size=(4, 4), activation="sigmoid", input_shape=(self.time_delay, self.channels)+self.image_size, data_format='channels_first', return_sequences=True))
        model.add(BatchNormalization())
        model.add(ConvLSTM2D(filters=10, kernel_size=(4, 4), activation="sigmoid")) #, input_shape=(self.time_delay, self.channels)+self.image_size, data_format='channels_first', return_sequences=True))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=1, kernel_size=(4, 4), activation="sigmoid", data_format="channels_first"))
        model.add(Flatten())
        model.add(Dense(units=len(self.target_labels), activation="sigmoid"))
        if self.verbose:
            model.summary()
        self.model = model

    def fit(self, features, labels, validation_split):
        """
        Trains the neural net on the data provided.

        :param features: Numpy array of training data.
        :param labels: Numpy array of target (label) data.
        :param validation_split: Float between 0 and 1. Percentage of training data to use for validation
        """
        self.model.compile(optimizer="RMSProp", loss="cosine_proximity", metrics=["accuracy"])
        self.model.fit(features, labels, batch_size=10, epochs=100, validation_split=validation_split,
            callbacks=[ReduceLROnPlateau(), EarlyStopping(patience=3)])
