import numpy as np
import random
from feature import Feature
from tdnn import TDNN

def prepare_training_data(sample_images):
    feature = Feature()

    x_temp = list()
    for image in sample_images:
        x_temp.append([[feature.extractFeatureVector(image)]])

    X_train = np.array(x_temp)
    return X_train

def prepare_training_labels(count):
    return np.array([random.uniform(0.8, 1)] * count)


def main():

    sample_images = list()
    for i in range(1,11):
        sample_images.append('../images/S502_001_000000%02d.png' % i)


    feature = Feature()
    X_train = prepare_training_data(sample_images)
    print (X_train.shape)
    y_train = prepare_training_labels(10)
    X_test = np.array([[[feature.extractFeatureVector('../images/S502_001_00000001.png')]]])
    print(X_test.shape)
    y_test = np.array([1])

    tdnn = TDNN(verbose=True)
    tdnn.train(X_train, y_train, X_test, y_test)


main()