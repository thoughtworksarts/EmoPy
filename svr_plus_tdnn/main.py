import sys
sys.path.append('../feature')
sys.path.append('../data')
from tdnn import TDNN
from data import get_image_feature_vector_array, get_training_label_array, get_time_delay_training_data
from regressionModel import RegressionModel
from dataProcessor import DataProcessor

def main(verbose=False):

    print("Extracting features...")
    d = DataProcessor()
    root_directory = "../data/cohn_kanade_images"

    d.add_feature('hog', {'orientations': 8, 'pixels_per_cell': (16, 16), 'cells_per_block': (1, 1)})
    # d.add_feature('lbp', {'n_points': 24, 'radius': 3})

    features = d.get_image_features(from_csv=False, dataset_location=root_directory, initial_image_dims=None, target_image_dims=(64,64), vector=True, time_series=False)

    labels = get_training_label_array()

    print("Training regression model...")
    model = RegressionModel(features, labels)
    model.fit()
    predictions = model.predict()

    print("Applying time-delay to regression output...")
    X_train, y_train, X_test, y_test = get_time_delay_training_data(predictions, predictions)

    if verbose:
        print ("X_train: " + str(X_train.shape))
        print ("y_train: " + str(y_train.shape))
        print("X_test: " + str(X_test.shape))
        print ("y_test: " + str(y_test.shape))

    print("Training TDNN...")
    tdnn = TDNN(verbose=True)
    tdnn.train(X_train, y_train, X_test, y_test)


main()