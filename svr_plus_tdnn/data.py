from feature import Feature
import os
import numpy as np
import math

EMOTION_DIMENSION_COUNT = 4 # emotional dimensions: arousal, valence, expectation, power


def get_image_feature_vector_array():
    feature = Feature()
    root_directory = "../data/cohn_kanade_images"
    features = list()
    for subfile in os.listdir(root_directory):
        if "DS_Store" in subfile:   continue
        for file in os.listdir(root_directory + "/" + subfile):
            if "DS_Store" not in file:
                image_file = root_directory + "/" + subfile + "/" + file
                features.append(feature.extract_hog_feature(image_file)[0])

    return np.array(features)

def get_raw_training_labels():
    # Uses 20 photo series from the Cohn-Kanade dataset
    # hand labeled by AP
    # arousal(least, most), valence(negative, positive), power, anticipation
    raw_training_labels = {1: [10, [.6, .4, .7, .6], [.9, .1, .8, .9]],
                           2: [9, [.2, .5, .6, .1], [.3, .4, .5, .2]],
                           3: [10, [.8, .9, .2, .9], [.99, .99, .1, .99]],
                           4: [10, [.2, .4, .4, .5], [.8, .2, .7, .6]],
                           5: [8, [.2, .4, .2, .1], [.5, .5, .5, .5]],
                           6: [10, [.8, .2, .2, .5], [.9, .1, .1, .5]],
                           7: [10, [.7, .4, .5, .6], [.8, .2, .8, .7]],
                           8: [9, [.5, .5, .4, .5], [.6, .4, .5, .3]],
                           9: [10, [.6, .4, .4, .7], [.9, .1, .1, .9]],
                           10: [10, [.1, .5, .2, .1], [.7, .2, .2, .5]],
                           11: [10, [.2, .4, .5, .2], [.3, .5, .4, .2]],
                           12: [10, [.6, .2, .2, .4], [.8, .1, .1, .4]],
                           13: [10, [.6, .4, .7, .5], [.8, .2, .8, .5]],
                           14: [9, [.1, .5, .5, .5], [.1, .4, .5, .4]],
                           15: [10, [.1, .4, .5, .5], [.8, .3, .4, .9]],
                           16: [10, [.6, .4, .7, .6], [.7, .2, .2, .5]],
                           17: [10, [.5, .4, .6, .5], [.7, .2, .7, .6]],
                           18: [10, [.7, .4, .2, .7], [.9, .1, .1, .9]],
                           19: [10, [.7, .3, .3, .5], [.8, .1, .1, .6]],
                           20: [10, [.6, .3, .2, .5], [.9, .1, .1, .6]]}
    labels = dict()
    for time_series_key in raw_training_labels:
        time_series = raw_training_labels[time_series_key]
        num_images = time_series[0]
        increment = [(time_series[2][emotion_dimension_idx] - time_series[1][emotion_dimension_idx]) / num_images for emotion_dimension_idx in range(EMOTION_DIMENSION_COUNT)]
        labels[time_series_key] = [[increment[label_idx]*image_idx + (time_series[1][label_idx]) for label_idx in range(EMOTION_DIMENSION_COUNT)] for image_idx in range(num_images)]

    return labels

def get_training_label_array():
    raw_training_labels = get_raw_training_labels()
    training_label_array = list()
    for time_series_key in raw_training_labels:
        time_series = raw_training_labels[time_series_key]
        training_label_array += time_series

    return np.array(training_label_array)

def get_time_delay_training_data(features, labels, time_delay=2, testing_percentage=0.25):
    X_train = list()
    for data_point_idx in range(time_delay, len(features)):
        data_point = [features[data_point_idx-offset] for offset in range(time_delay+1)]
        X_train.append([data_point])

    y_train = labels[time_delay:len(labels)]

    X_test = np.array(X_train[int(math.ceil(len(X_train)*(1-testing_percentage))):len(X_train)])
    X_train = np.array(X_train[0:int(math.ceil(len(X_train)*(1-testing_percentage)))])
    y_test = np.array(y_train[int(math.ceil(len(y_train)*(1-testing_percentage))):len(y_train)])
    y_train = np.array(y_train[0:int(math.ceil(len(y_train)*(1-testing_percentage)))])

    return (X_train, y_train, X_test, y_test)



# -------------------------------- UNUSED -------------------------------------- #

def get_time_delay_image_training_data(time_delay=2):
    features = get_image_feature_vector_batches()

    X_train = list()
    for time_series_idx in features:
        vec = features[time_series_idx]
        for data_idx in range(time_delay, len(vec)):
            X_train.append([[vec[j] for j in range(data_idx-time_delay, data_idx+1)]])

    return np.array(X_train)

def get_image_feature_vector_batches():
    feature = Feature()
    root_directory = "../data/cohn_kanade_images"
    features = dict()
    idx = 1
    for subfile in os.listdir(root_directory):
        if "DS_Store" in subfile:   continue
        subfeatures = list()
        for file in os.listdir(root_directory + "/" + subfile):
            if "DS_Store" not in file:
                image_file = root_directory + "/" + subfile + "/" + file
                subfeatures.append(feature.extract_hog_feature(image_file)[1])
        features[idx] = subfeatures
        idx += 1

    return features


def get_shifted_training_labels(time_delay=2):
    raw_training_labels = get_raw_training_labels()
    shifted_training_labels = list()
    for batch_index in raw_training_labels:
        labels_in_batch = raw_training_labels[batch_index]
        shifted_training_labels.append(labels_in_batch[time_delay:len(labels_in_batch)])

    final_labels = list()
    for shifted_label in shifted_training_labels:
        final_labels += shifted_label

    return np.array(final_labels)

def get_delayed_emotion_training_data(time_delay=2):
    raw_training_labels = get_raw_training_labels()
    X_train, y_train = get_delayed_training_data(time_delay, range(1,18), raw_training_labels)
    X_test, y_test = get_delayed_training_data(time_delay, range(18,21), raw_training_labels)

    return (X_train, y_train, X_test, y_test)

def get_delayed_training_data(time_delay, time_series_keys, raw_training_data):
    X = list()
    y = list()
    for time_series_idx in time_series_keys:
        time_series = raw_training_data[time_series_idx]
        for image_idx in range(time_delay, len(time_series)):
            data_point = [time_series[time_delay_idx] for time_delay_idx in range(image_idx-time_delay, image_idx)]
            label = time_series[image_idx]
            X.append([data_point])
            y.append(label)

    return (np.array(X), np.array(y))
