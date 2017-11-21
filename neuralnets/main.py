import sys
sys.path.append('../feature')
sys.path.append('../data')
sys.path.append('../svr_plus_tdnn') # TODO: temporary
from imageprocessor import ImageProcessor
from transfer_model import TransferModel
from tdnn import TDNN
from regressionModel import RegressionModel
from featureextractor import FeatureExtractor

runInceptionV3 = False
runRegressionPlusTDNN = True
runConvLSTM = False

verbose = True
target_dimensions = (128, 128)
raw_dimensions = (48, 48)

if runInceptionV3:
    print('--------------- Inception-V3 Model -------------------')
    print('Creating NN with InceptionV3 base model...')
    model = TransferModel(model_name='inception_v3')

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


if runRegressionPlusTDNN:

    print('--------------- Regression + TDNN Model -------------------')
    print('Collecting data...')
    root_directory = '../data/cohn_kanade_images'
    imageProcessor = ImageProcessor(from_csv=False, datapath=root_directory, target_dimensions=target_dimensions, raw_dimensions=None)
    images, labels = imageProcessor.get_training_data()

    print ('images shape: ' + str(images.shape))
    print('Extracting features...')

    featureExtractor = FeatureExtractor(images, return_array=False)
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

    print('Training TDNN...')
    tdnn = TDNN(verbose=True)
    tdnn.train(X_train, y_train, X_test, y_test)



