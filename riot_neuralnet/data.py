from feature import Feature
import os
import numpy as np

def get_time_delay_training_data(time_delay=2):
    features = get_feature_vectors()

    X_train = list()
    for timeSeriesIdx in features:
        vec = features[timeSeriesIdx]
        for dataIdx in range(time_delay, len(vec)):
            X_train.append([[vec[j] for j in range(dataIdx-time_delay, dataIdx+1)]])

    return np.array(X_train)

def get_feature_vectors():
    feature = Feature()
    # iterate in directory
    rootdir = '../images'
    features = dict()
    idx = 1
    for subfile in os.listdir(rootdir):
        subfeatures = list()
        if "DS_Store" in subfile:   continue
        for file in os.listdir(rootdir + '/' + subfile):
            if "DS_Store" not in file:
                imageFile = rootdir + '/' + subfile + '/' + file
                subfeatures.append(feature.extractFeatureVector(imageFile))
        features[idx] = subfeatures
        idx += 1

    return features


def get_shifted_training_labels(time_delay=2):
    raw_training_labels = get_raw_training_labels()
    shifted_training_labels = list()
    for batch_index in raw_training_labels:
        labels_in_batch = raw_training_labels[batch_index]
        shifted_training_labels.append(labels_in_batch[time_delay:len(labels_in_batch)])

    finalLabels = list()
    for shifted_label in shifted_training_labels:
        finalLabels += shifted_label

    return np.array(finalLabels)

def get_delayed_emotion_training_data(time_delay=2):
    raw_training_labels = get_raw_training_labels()
    X_train, y_train = get_delayed_training_data(time_delay, range(1,9), raw_training_labels)
    X_test, y_test = get_delayed_training_data(time_delay, range(9,11), raw_training_labels)

    return (X_train, y_train, X_test, y_test)

def get_delayed_training_data(time_delay, keys, raw_training_data):
    X = list()
    y = list()
    for time_series_idx in keys:
        time_series = raw_training_data[time_series_idx]
        for image_idx in range(time_delay, len(time_series)):
            data_point = [time_series[time_delay_idx] for time_delay_idx in range(image_idx-time_delay, image_idx)]
            label = time_series[image_idx]
            X.append([data_point])
            y.append(label)

    return (np.array(X), np.array(y))

def get_raw_training_labels():
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
        labels[key] = rowLabels

    return labels