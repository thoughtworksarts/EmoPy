import sys
sys.path.append('../data')
# sys.path.append('../svr_plus_tdnn') # TODO: remove once regression step is moved into TimeDelayNN class
from imageprocessor import ImageProcessor
from neuralnets import TransferLearningNN, TimeDelayNN, ConvolutionalLstmNN
from regression_model import RegressionModel
from featureextractor import FeatureExtractor
import numpy as np
import math

runTransferLearningNN = True
runRegressionPlusTimeDelayNN = True
runConvLSTM = True

verbose = True
target_dimensions = (128, 128)
raw_dimensions = (48, 48)

if runTransferLearningNN:
    print('--------------- Inception-V3 Model -------------------')
    print('Creating NN with InceptionV3 base model...')
    model = TransferLearningNN(model_name='inception_v3')

    print('Extracting training data...')


    csv_file_path = "../data/fer2013/fer2013.csv"
    root_directory = "../data/cohn_kanade_images"
    imageProcessor = ImageProcessor(from_csv=True, datapath=csv_file_path, target_dimensions=target_dimensions, raw_dimensions=raw_dimensions, csv_label_col=0, csv_image_col=1)

    # imageProcessor.add_feature('hog', {'orientations': 8, 'pixels_per_cell': (4, 4), 'cells_per_block': (1, 1)})

    X_train, y_train, X_test, y_test = imageProcessor.get_training_data()

    print('X_train shape: ' + str(X_train.shape))
    print('y_train shape: ' + str(y_train.shape))
    print('X_test shape: ' + str(X_test.shape))
    print('y_test shape: ' + str(y_test.shape))


    print ('Training model...')
    print('numLayers: ' + str(len(model.model.layers)))
    model.fit(X_train, y_train, X_test, y_test)

    trained_model_output_filepath = '../trained_models/inception_v3_model_1.h5'
    model.model.save(trained_model_output_filepath)


if runRegressionPlusTimeDelayNN:

    print('--------------- Regression + TimeDelayNN Model -------------------')
    print('Collecting data...')
    root_directory = '../data/cohn_kanade_images'
    imageProcessor = ImageProcessor(from_csv=False, datapath=root_directory, target_dimensions=target_dimensions, raw_dimensions=None)
    images, labels = imageProcessor.get_training_data()

    print ('images shape: ' + str(images.shape))
    print('Extracting features...')
    featureExtractor = FeatureExtractor(images, return_2d_array=False)
    featureExtractor.add_feature('hog', {'orientations': 8, 'pixels_per_cell': (16, 16), 'cells_per_block': (1, 1)})
    # featureExtractor.add_feature('lbp', {'n_points': 24, 'radius': 3})
    features = featureExtractor.extract()
    print ("features shape: " + str(features.shape))

    print('Training regression model...')
    model = RegressionModel(features, labels)
    model.fit()
    predictions = model.predict()

    print('Applying time-delay to regression output...')
    X_train, y_train, X_test, y_test = imageProcessor.get_time_delay_training_data(predictions, predictions)
    if verbose:
        print ('X_train: ' + str(X_train.shape))
        print ('y_train: ' + str(y_train.shape))
        print('X_test: ' + str(X_test.shape))
        print ('y_test: ' + str(y_test.shape))

    print('Training TimeDelayNN...')
    tdnn = TimeDelayNN(verbose=True)
    tdnn.fit(X_train, y_train, X_test, y_test)

if runConvLSTM:
    samples = None
    time_delay = 1
    target_dimensions = (64, 64)
    channels = 1
    verbose = True

    print('--------------- Convolutional LSTM Model -------------------')
    print('Collecting data...')
    root_directory = "../data/cohn_kanade_images"
    imageProcessor = ImageProcessor(from_csv=False, datapath=root_directory, target_dimensions=target_dimensions, raw_dimensions=raw_dimensions)
    images, labels = imageProcessor.get_training_data()

    print ('images shape: ' + str(images.shape))
    print('Extracting features...')
    featureExtractor = FeatureExtractor(images, return_2d_array=True)
    featureExtractor.add_feature('hog', {'orientations': 8, 'pixels_per_cell': (16, 16), 'cells_per_block': (1, 1)})
    # featureExtractor.add_feature('lbp', {'n_points': 24, 'radius': 3})
    raw_features = featureExtractor.extract()
    features = list()
    for feature in raw_features:
        features.append([[feature]])
    features = np.array(features)
    if verbose:
        print('feature shape: ' + str(features.shape))
        print('label shape: ' + str(labels.shape))

    print('Creating training/testing data...')
    testing_percentage = 0.20
    X_test = np.array(features[int(math.ceil(len(features)*(1-testing_percentage))):len(features)])
    X_train = np.array(features[0:int(math.ceil(len(features)*(1-testing_percentage)))])
    y_test = np.array(labels[int(math.ceil(len(labels)*(1-testing_percentage))):len(labels)])
    y_train = np.array(labels[0:int(math.ceil(len(labels)*(1-testing_percentage)))])
    if verbose:
        print('X_train shape: ' + str(X_train.shape))
        print('y_train shape: ' + str(y_train.shape))
        print('X_test shape: ' + str(X_test.shape))
        print('y_test shape: ' + str(y_test.shape))

    print('Training net...')
    net = ConvolutionalLstmNN(target_dimensions, channels, time_delay=time_delay)
    net.fit(X_train, y_train, X_test, y_test)

