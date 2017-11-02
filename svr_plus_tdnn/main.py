from tdnn import TDNN
from data import get_image_feature_vector_array, get_training_label_array, get_time_delay_training_data
from regressionModel import RegressionModel


def main(verbose=False):

    print "Extracting features..."
    features = get_image_feature_vector_array()
    labels = get_training_label_array()

    print "Training regression model..."
    model = RegressionModel(features, labels)
    model.fit()
    predictions = model.predict()

    print "Applying time-delay to regression output..."
    X_train, y_train, X_test, y_test = get_time_delay_training_data(predictions, predictions)

    if verbose:
        print ("X_train: " + str(X_train.shape))
        print ("y_train: " + str(y_train.shape))
        print("X_test: " + str(X_test.shape))
        print ("y_test: " + str(y_test.shape))

    print "Training TDNN..."
    tdnn = TDNN(verbose=True)
    tdnn.train(X_train, y_train, X_test, y_test)


main()