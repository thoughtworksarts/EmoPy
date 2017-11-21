import sys
sys.path.append('../feature')
sys.path.append('../data')
sys.path.append('../svr_plus_tdnn') # TODO: temporary
from dataProcessor import DataProcessor
from transfer_model import TransferModel
from tdnn import TDNN
from regressionModel import RegressionModel

runInceptionV3 = False
runRegressionPlusTDNN = True
runConvLSTM = False

if runInceptionV3:
    print('--------------- Inception-V3 Model -------------------')
    print('Creating NN with InceptionV3 base model...')
    model = TransferModel(model_name='inception_v3')

    print('Extracting training data...')

    target_image_dims = (128,128)

    d = DataProcessor()
    root_directory = "../data/cohn_kanade_images"
    csv_file_path = "../data/fer2013/fer2013.csv"

    d.add_feature('hog', {'orientations': 8, 'pixels_per_cell': (4, 4), 'cells_per_block': (1, 1)})

    X_train, y_train, X_test, y_test = d.get_training_data(from_csv=True, dataset_location=csv_file_path, target_image_dims=target_image_dims, initial_image_dims=(48, 48), label_index=0, image_index=1, vector=False, time_series=False)

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
    print('Extracting features...')
    d = DataProcessor()
    root_directory = '../data/cohn_kanade_images'

    d.add_feature('hog', {'orientations': 8, 'pixels_per_cell': (16, 16), 'cells_per_block': (1, 1)})
    # d.add_feature('lbp', {'n_points': 24, 'radius': 3})

if False:
    features = d.get_training_data(from_csv=False, dataset_location=root_directory, initial_image_dims=None, target_image_dims=(64, 64), vector=True, time_series=False)

    # TODO: Add label processing to DataProcessor class
    labels = d.get_training_label_array()

    print('Training regression model...')
    model = RegressionModel(features, labels)
    model.fit()
    predictions = model.predict()

    print('Applying time-delay to regression output...')
    X_train, y_train, X_test, y_test = get_time_delay_training_data(predictions, predictions)

    if verbose:
        print ('X_train: ' + str(X_train.shape))
        print ('y_train: ' + str(y_train.shape))
        print('X_test: ' + str(X_test.shape))
        print ('y_test: ' + str(y_test.shape))

    print('Training TDNN...')
    tdnn = TDNN(verbose=True)
    tdnn.train(X_train, y_train, X_test, y_test)



