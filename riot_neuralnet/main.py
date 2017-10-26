import numpy as np
import random
from feature import Feature
from tdnn import TDNN
import os

def prepare_training_data():
    feature = Feature()

    # iterate in directory
    rootdir = '../images'
    X_train = list()
    for subfile in os.listdir(rootdir):
        if "DS_Store" in subfile:   continue
        for file in os.listdir(rootdir + '/' + subfile):
            if "DS_Store" not in file:
                imageFile = rootdir + '/' + subfile + '/' + file
                X_train.append([[feature.extractFeatureVector(imageFile)]])

    return np.array(X_train)


def prepare_training_labels():
    # Uses 10 photo series from the Cohn-Kanade dataset
    # arousal(least, most), valence(negative, positive), power, anticipation
    prelabels = {1: [10, [.6, .4, .7, .6], [.9, .1, .8, .9]],
                 2: [9, [.2, .5, .6, .1], [.3, .4, .5, .2]],
                 3: [10, [.8, .9, .2, .9], [.99, .99, .1, .99]],
                 4: [10, [.2, .4, .4, .5], [.8, .2, .7, .6]],
                 5: [8, [.2, .4, .2, .1], [.5, .5, .5, .5]],
                 6: [10, [.8, .2, .2, .5], [.9, .1, .1, .5]],
                 7: [10, [.7, .4, .5, .6], [.8, .2, .8, .7]],
                 8: [9, [.5, .5, .4, .5], [.6, .4, .5, .3]],
                 9: [10, [.6, .4, .4, .7], [.9, .1, .1, .9]],
                 10: [10, [.1, .5, .2, .1], [.7, .2, .2, .5]]
                 }

    labels = dict()

    for i, key in enumerate(prelabels):
        row = prelabels[key]
        numImages = row[0]

        increment = list()
        for j in range(4):
            increment.append((row[2][j] - row[1][j]) / numImages)

        rowLabels = list()
        for k in range(numImages):
            newLabel = list()
            for labelIdx in range(4):
                newLabel.append(increment[labelIdx]*k+(row[1][labelIdx]))
            rowLabels.append(newLabel)
        labels[i] = rowLabels

    finalLabels = list()
    for key in labels:
        finalLabels += labels[key]

    return np.array(finalLabels)

def main():
    feature = Feature()
    X_train = prepare_training_data()
    print ('X_train: ' + str(X_train.shape))
    y_train = prepare_training_labels()
    print ('y_train: ' + str(y_train.shape))
    X_test = np.array([[[feature.extractFeatureVector('../images/001/S502_001_00000001.png')]]])
    print('X_test: ' + str(X_test.shape))
    y_test = np.array([[.6, .4, .7, .6]])
    print ('y_test: ' + str(y_test.shape))

    tdnn = TDNN(verbose=True)
    tdnn.train(X_train, y_train, X_test, y_test)


main()