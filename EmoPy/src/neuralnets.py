from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.layers import Dense, Flatten, GlobalAveragePooling2D, Conv2D, ConvLSTM2D, Conv3D, MaxPooling2D, Dropout, \
    MaxPooling3D
from keras.layers.normalization import BatchNormalization
from keras.models import Model, Sequential
from keras.utils import plot_model
import json

from EmoPy.src.callback import PlotLosses


class _FERNeuralNet(object):
    """
    Interface for all FER deep neural net classes.
    """

    def __init__(self, emotion_map):
        self.emotion_map = emotion_map
        self._init_model()

    def _init_model(self):
        raise NotImplementedError("Class %s doesn't implement _init_model()" % self.__class__.__name__)

    def fit(self, x_train, y_train):
        raise NotImplementedError("Class %s doesn't implement fit()" % self.__class__.__name__)

    def fit_generator(self, generator, validation_data=None, epochs=50):
        self.model.compile(optimizer="RMSProp", loss="cosine_proximity", metrics=["accuracy"])
        self.model.fit_generator(generator=generator, validation_data=validation_data, epochs=epochs,
                                 callbacks=[ReduceLROnPlateau(), EarlyStopping(patience=3), PlotLosses()])

    def predict(self, images):
        self.model.predict(images)

    def save_model_graph(self):
        plot_model(self.model, to_file='output/model.png')

    def export_model(self, model_filepath, weights_filepath, emotion_map_filepath, emotion_map):
        self.model.save_weights(weights_filepath)

        model_json_string = self.model.to_json()
        model_json_file = open(model_filepath, 'w')
        model_json_file.write(model_json_string)
        model_json_file.close()

        with open(emotion_map_filepath, 'w') as fp:
            json.dump(emotion_map, fp)


class TransferLearningNN(_FERNeuralNet):
    """
    Transfer Learning Convolutional Neural Network initialized with pretrained weights.

    :param model_name: name of pretrained model to use for initial weights. Options: ['Xception', 'VGG16', 'VGG19', 'ResNet50', 'InceptionV3', 'InceptionResNetV2']
    :param emotion_map: dict of target emotion label keys with int values corresponding to the index of the emotion probability in the prediction output array

    **Example**::

        model = TransferLearningNN(model_name='inception_v3', target_labels=[0,1,2,3,4,5,6])
        model.fit(images, labels, validation_split=0.15)

    """
    _NUM_BOTTOM_LAYERS_TO_RETRAIN = 249

    def __init__(self, model_name, emotion_map):
        self.model_name = model_name
        super().__init__(emotion_map)

    def _init_model(self):
        """
        Initialize base model from Keras and add top layers to match number of training emotions labels.
        :return:
        """
        base_model = self._get_base_model()

        top_layer_model = base_model.output
        top_layer_model = GlobalAveragePooling2D()(top_layer_model)
        top_layer_model = Dense(1024, activation='relu')(top_layer_model)
        prediction_layer = Dense(output_dim=len(self.emotion_map.keys()), activation='softmax')(top_layer_model)

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
        self.model.fit(x=features, y=labels, epochs=epochs, verbose=1,
                       callbacks=[ReduceLROnPlateau(), EarlyStopping(patience=3)], validation_split=validation_split,
                       shuffle=True)

        for layer in self.model.layers[:self._NUM_BOTTOM_LAYERS_TO_RETRAIN]:
            layer.trainable = False
        for layer in self.model.layers[self._NUM_BOTTOM_LAYERS_TO_RETRAIN:]:
            layer.trainable = True

        self.model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(x=features, y=labels, epochs=50, verbose=1,
                       callbacks=[ReduceLROnPlateau(), EarlyStopping(patience=3)], validation_split=validation_split,
                       shuffle=True)

class ConvolutionalLstmNN(_FERNeuralNet):
    """
    Convolutional Long Short Term Memory Neural Network.

    :param image_size: dimensions of input images
    :param channels: number of image channels
    :param emotion_map: dict of target emotion label keys with int values corresponding to the index of the emotion probability in the prediction output array
    :param time_delay: number of time steps for lookback
    :param filters: number of filters/nodes per layer in CNN
    :param kernel_size: size of sliding window for each layer of CNN
    :param activation: name of activation function for CNN
    :param verbose: if true, will print out extra process information

    **Example**::

        net = ConvolutionalLstmNN(target_dimensions=(64,64), channels=1, target_labels=[0,1,2,3,4,5,6], time_delay=3)
        net.fit(features, labels, validation_split=0.15)

    """

    def __init__(self, image_size, channels, emotion_map, time_delay=2, filters=10, kernel_size=(4, 4),
                 activation='sigmoid', verbose=False):
        self.time_delay = time_delay
        self.channels = channels
        self.image_size = image_size
        self.verbose = verbose

        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation
        super().__init__(emotion_map)

    def _init_model(self):
        """
        Composes all layers of CNN.
        """
        model = Sequential()
        model.add(ConvLSTM2D(filters=self.filters, kernel_size=self.kernel_size, activation=self.activation,
                             input_shape=[self.time_delay] + list(self.image_size) + [self.channels],
                             data_format='channels_last', return_sequences=True))
        model.add(BatchNormalization())
        model.add(ConvLSTM2D(filters=self.filters, kernel_size=self.kernel_size, activation=self.activation,
                             input_shape=(self.time_delay, self.channels) + self.image_size,
                             data_format='channels_last', return_sequences=True))
        model.add(BatchNormalization())
        model.add(ConvLSTM2D(filters=self.filters, kernel_size=self.kernel_size, activation=self.activation))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=1, kernel_size=self.kernel_size, activation="sigmoid", data_format="channels_last"))
        model.add(Flatten())
        model.add(Dense(units=len(self.emotion_map.keys()), activation="sigmoid"))
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

class ConvolutionalNN(_FERNeuralNet):
    """
    2D Convolutional Neural Network

    :param image_size: dimensions of input images
    :param channels: number of image channels
    :param emotion_map: dict of target emotion label keys with int values corresponding to the index of the emotion probability in the prediction output array
    :param filters: number of filters/nodes per layer in CNN
    :param kernel_size: size of sliding window for each layer of CNN
    :param activation: name of activation function for CNN
    :param verbose: if true, will print out extra process information

    **Example**::

        net = ConvolutionalNN(target_dimensions=(64,64), channels=1, target_labels=[0,1,2,3,4,5,6], time_delay=3)
        net.fit(features, labels, validation_split=0.15)

    """

    def __init__(self, image_size, channels, emotion_map, filters=10, kernel_size=(4, 4), activation='relu',
                 verbose=False):
        self.channels = channels
        self.image_size = image_size
        self.verbose = verbose

        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation
        super().__init__(emotion_map)

    def _init_model(self):
        """
        Composes all layers of 2D CNN.
        """
        model = Sequential()
        model.add(Conv2D(input_shape=list(self.image_size) + [self.channels], filters=self.filters,
                         kernel_size=self.kernel_size, activation='relu', data_format='channels_last'))
        model.add(
            Conv2D(filters=self.filters, kernel_size=self.kernel_size, activation='relu', data_format='channels_last'))
        model.add(MaxPooling2D())
        model.add(
            Conv2D(filters=self.filters, kernel_size=self.kernel_size, activation='relu', data_format='channels_last'))
        model.add(
            Conv2D(filters=self.filters, kernel_size=self.kernel_size, activation='relu', data_format='channels_last'))
        model.add(MaxPooling2D())

        model.add(Flatten())
        model.add(Dense(units=len(self.emotion_map.keys()), activation="relu"))
        if self.verbose:
            model.summary()
        self.model = model

    def fit(self, image_data, labels, validation_split, epochs=50):
        """
        Trains the neural net on the data provided.

        :param image_data: Numpy array of training data.
        :param labels: Numpy array of target (label) data.
        :param validation_split: Float between 0 and 1. Percentage of training data to use for validation
        :param batch_size:
        :param epochs: number of times to train over input dataset.
        """
        self.model.compile(optimizer="RMSProp", loss="cosine_proximity", metrics=["accuracy"])
        self.model.fit(image_data, labels, epochs=epochs, validation_split=validation_split,
                       callbacks=[ReduceLROnPlateau(), EarlyStopping(patience=3)])

class TimeDelayConvNN(_FERNeuralNet):
    """
    The Time-Delayed Convolutional Neural Network model is a 3D-Convolutional network that trains on 3-dimensional temporal image data. One training sample will contain n number of images from a series and its emotion label will be that of the most recent image.

    :param image_size: dimensions of input images
    :param time_delay: number of past time steps included in each training sample
    :param channels: number of image channels
    :param emotion_map: dict of target emotion label keys with int values corresponding to the index of the emotion probability in the prediction output array
    :param filters: number of filters/nodes per layer in CNN
    :param kernel_size: size of sliding window for each layer of CNN
    :param activation: name of activation function for CNN
    :param verbose: if true, will print out extra process information

    **Example**::

        model = TimeDelayConvNN(target_dimensions={64,64), time_delay=3, channels=1, label_count=6)
        model.fit(image_data, labels, validation_split=0.15)

    """

    def __init__(self, image_size, channels, emotion_map, time_delay, filters=32, kernel_size=(1, 4, 4),
                 activation='relu', verbose=False):
        self.image_size = image_size
        self.time_delay = time_delay
        self.channels = channels
        self.verbose = verbose

        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation
        super().__init__(emotion_map)

    def _init_model(self):
        """
        Composes all layers of 3D CNN.
        """
        model = Sequential()
        model.add(Conv3D(input_shape=[self.time_delay] + list(self.image_size) + [self.channels], filters=self.filters,
                         kernel_size=self.kernel_size, activation='relu', data_format='channels_last'))
        model.add(
            Conv3D(filters=self.filters, kernel_size=self.kernel_size, activation='relu', data_format='channels_last'))
        model.add(MaxPooling3D(pool_size=(1, 2, 2), data_format='channels_last'))
        model.add(
            Conv3D(filters=self.filters, kernel_size=self.kernel_size, activation='relu', data_format='channels_last'))
        model.add(
            Conv3D(filters=self.filters, kernel_size=self.kernel_size, activation='relu', data_format='channels_last'))
        model.add(MaxPooling3D(pool_size=(1, 2, 2), data_format='channels_last'))

        model.add(Flatten())
        model.add(Dense(units=len(self.emotion_map.keys()), activation="relu"))
        if self.verbose:
            model.summary()
        self.model = model

    def fit(self, image_data, labels, validation_split, epochs=50):
        """
        Trains the neural net on the data provided.

        :param image_data: Numpy array of training data.
        :param labels: Numpy array of target (label) data.
        :param validation_split: Float between 0 and 1. Percentage of training data to use for validation
        :param batch_size:
        :param epochs: number of times to train over input dataset.
        """
        self.model.compile(optimizer="RMSProp", loss="cosine_proximity", metrics=["accuracy"])
        self.model.fit(image_data, labels, epochs=epochs, validation_split=validation_split,
                       callbacks=[ReduceLROnPlateau(), EarlyStopping(patience=3)])
