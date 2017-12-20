from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
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

    def fit(self, x_train, y_train):
        raise NotImplementedError("Class %s doesn't implement fit()" % self.__class__.__name__)

    def predict(self, images):
        raise NotImplementedError("Class %s doesn't implement predict()" % self.__class__.__name__)


class TransferLearningNN(_FERNeuralNet):
    """
    Transfer Learning Convolutional Neural Network initialized with pretrained weights.

    :param model_name: name of pretrained model to use for initial weights. Options: ['Xception', 'VGG16', 'VGG19', 'ResNet50', 'InceptionV3', 'InceptionResNetV2']
    :param target_labels: list of target emotion labels

    **Example**::

        csv_file_path = '<local csv file path>'
        target_labels = [0,1,2,3,4,5,6]

        imageProcessor = ImageProcessor(from_csv=True, target_labels=target_labels, datapath=csv_file_path, target_dimensions=(64,64), raw_dimensions=(48,48), csv_label_col=0, csv_image_col=1)
        features, labels = imageProcessor.get_training_data()

        model = TransferLearningNN(model_name='inception_v3', target_labels=target_labels)
        model.fit(features, labels, 0.15)

    """
    _NUM_BOTTOM_LAYERS_TO_RETRAIN = 249

    def __init__(self, model_name, target_labels):
        self.model_name = model_name
        self.target_labels = target_labels
        super().__init__()

    def _init_model(self):
        """
        Initialize base model from Keras and add top layers to match number of training emotions labels.
        :return:
        """
        base_model = self._get_base_model()

        top_layer_model = base_model.output
        top_layer_model = GlobalAveragePooling2D()(top_layer_model)
        top_layer_model = Dense(1024, activation='relu')(top_layer_model)
        prediction_layer = Dense(output_dim=len(self.target_labels), activation='softmax')(top_layer_model)

        model = Model(input=base_model.input, output=prediction_layer)
        print(model.summary())
        for layer in base_model.layers:
            layer.trainable = False
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

        self.model = model

    def _get_base_model(self):
        """
        :return: base model from Keras based on user-supplied model name
        """
        if self.model_name == 'inception_v3':
            return InceptionV3(weights='imagenet', include_top=False)
        elif self.model_name == 'xception':
            return Xception(weights='imagenet', include_top=False)
        elif self.model_name == 'vgg16':
            return VGG16(weights='imagenet', include_top=False)
        elif self.model_name == 'vgg19':
            return VGG19(weights='imagenet', include_top=False)
        elif self.model_name == 'resnet50':
            return ResNet50(weights='imagenet', include_top=False)
        else:
            raise ValueError('Cannot find base model %s' % self.model_name)

    def fit(self, features, labels, validation_split, epochs=50):
        """
        Trains the neural net on the data provided.

        :param features: Numpy array of training data.
        :param labels: Numpy array of target (label) data.
        :param validation_split: Float between 0 and 1. Percentage of training data to use for validation
        :param epochs: Max number of times to train over dataset.
        """
        self.model.fit(x=features, y=labels, epochs=epochs, verbose=1, callbacks=[ReduceLROnPlateau(), EarlyStopping(patience=3)], validation_split=validation_split, shuffle=True)

        for layer in self.model.layers[:self._NUM_BOTTOM_LAYERS_TO_RETRAIN]:
            layer.trainable = False
        for layer in self.model.layers[self._NUM_BOTTOM_LAYERS_TO_RETRAIN:]:
            layer.trainable = True

        self.model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(x=features, y=labels, epochs=50, verbose=1, callbacks=[ReduceLROnPlateau(), EarlyStopping(patience=3)], validation_split=validation_split, shuffle=True)

    def predict(self, images):
        self.model.predict(images)


class TimeDelayNN(_FERNeuralNet):
    """
    This model is broken down into two steps: regression and CNN variant.
    The regression step is trained on the supplied image dataset, and its output is used to train the CNN. The output from the regression step is preprocessed to create new "time-delayed" datapoints. In other words, the output becomes 3-dimensional (1, regression_output_dim, number_of_previous_time_steps_considered +1)

    :param feature_vector_length: length of input feature vectors used to train regression step
    :param time_delay: number of previous datapoints to consider for each new datapoint in CNN step
    :param num_output_values: number of classification labels
    :param verbose: if true, will print out extra process information

    **Example**::

        target_dimensions = (128, 128)
        target_labels = [0,1,2,3]

        root_directory = '<local training image directory>'
        imageProcessor = ImageProcessor(from_csv=False, target_labels=target_labels, datapath=root_directory, target_dimensions=target_dimensions)
        images, labels = imageProcessor.get_training_data()

        featureExtractor = FeatureExtractor(images, return_2d_array=False)
        featureExtractor.add_feature('hog', {'orientations': 8, 'pixels_per_cell': (16, 16), 'cells_per_block': (1, 1)})
        features = featureExtractor.extract()

        tdnn = TimeDelayNN(len(features[0]), verbose=True)
        tdnn.fit(features, labels, 0.15)

    """
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
        model.add(Dense(units=self.num_output_values, activation="sigmoid"))
        if self.verbose:
            model.summary()
        self.model = model

    def fit(self, features, labels, validation_split, batch_size=10, epochs=20):
        """
        Trains the neural net on the data provided.

        :param features: Numpy array of training data.
        :param labels: Numpy array of target (label) data.
        :param validation_split: Float between 0 and 1. Percentage of training data to use for validation
        :param batch_size:
        :param epochs: Max number of times to train on input data.
        """
        self.regression_model.compile(loss="mean_squared_error", optimizer="rmsprop", metrics=["accuracy"])
        self.regression_model.fit(features, labels,
                       batch_size=batch_size, epochs=epochs,
                       validation_split=0.0, shuffle=True)

        regression_predictions = self.regression_model.predict(features)
        imageProcessor = ImageProcessor()
        features, labels = imageProcessor._get_time_delay_training_data(regression_predictions, regression_predictions)
        self.model.compile(optimizer="RMSProp", loss="cosine_proximity", metrics=["accuracy"])
        self.model.fit(features, labels, batch_size=batch_size, epochs=epochs, validation_split=validation_split, callbacks=[ReduceLROnPlateau(), EarlyStopping(patience=3)])

    def predict(self, images):
        pass


class ConvolutionalLstmNN(_FERNeuralNet):
    """
    Convolutional Long Short Term Memory Neural Network.

    :param image_size: dimensions of input images
    :param channels: number of image channels
    :param target_labels: list of target emotion labels
    :param time_delay: number time steps for lookback
    :param filters: number of filters/nodes per layer in CNN
    :param kernel_size: size of sliding window for each layer of CNN
    :param activation: name of activation function for CNN
    :param verbose: if true, will print out extra process information

    **Example**::

        target_labels = [0,1,2,3,4,5,6]
        csv_file_path = "<local csv file path>"

        imageProcessor = ImageProcessor(from_csv=True, target_labels=target_labels, datapath=csv_file_path, target_dimensions=(64,64), raw_dimensions=(48,48), csv_label_col=0, csv_image_col=1, channels=1)
        images, labels = imageProcessor.get_training_data()

        featureExtractor = FeatureExtractor(images, return_2d_array=True)
        featureExtractor.add_feature('hog', {'orientations': 8, 'pixels_per_cell': (16, 16), 'cells_per_block': (1, 1)})
        features = featureExtractor.extract()

        net = ConvolutionalLstmNN(target_dimensions=(64,64), channels=1, target_labels=target_labels, time_delay=3)
        net.fit(features, labels, validation_split=0.15)

    """

    def __init__(self, image_size, channels, target_labels, time_delay=2, filters=10, kernel_size=(4,4), activation='sigmoid', verbose=False):
        self.time_delay = time_delay
        self.channels = channels
        self.image_size = image_size
        self.target_labels = target_labels
        self.verbose = verbose

        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation
        super().__init__()

    def _init_model(self):
        """
        Composes all layers of CNN.
        """
        model = Sequential()
        model.add(ConvLSTM2D(filters=self.filters, kernel_size=self.kernel_size, activation=self.activation, input_shape=(self.time_delay, self.channels)+self.image_size, data_format='channels_first', return_sequences=True))
        model.add(BatchNormalization())
        model.add(ConvLSTM2D(filters=self.filters, kernel_size=self.kernel_size, activation=self.activation, input_shape=(self.time_delay, self.channels)+self.image_size, data_format='channels_first', return_sequences=True))
        model.add(BatchNormalization())
        model.add(ConvLSTM2D(filters=self.filters, kernel_size=self.kernel_size, activation=self.activation))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=1, kernel_size=self.kernel_size, activation="sigmoid", data_format="channels_first"))
        model.add(Flatten())
        model.add(Dense(units=len(self.target_labels), activation="sigmoid"))
        if self.verbose:
            model.summary()
        self.model = model

    def fit(self, features, labels, validation_split, batch_size=10, epochs=50):
        """
        Trains the neural net on the data provided.

        :param features: Numpy array of training data.
        :param labels: Numpy array of target (label) data.
        :param validation_split: Float between 0 and 1. Percentage of training data to use for validation
        :param batch_size:
        :param epochs: number of times to train over input dataset.
        """
        self.model.compile(optimizer="RMSProp", loss="cosine_proximity", metrics=["accuracy"])
        self.model.fit(features, labels, batch_size=batch_size, epochs=epochs, validation_split=validation_split,
            callbacks=[ReduceLROnPlateau(), EarlyStopping(patience=3)])

    def predict(self, images):
        self.model.predict(images)

