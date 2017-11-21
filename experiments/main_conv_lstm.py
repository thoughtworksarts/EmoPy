import sys
sys.path.append('../feature')
sys.path.append('../data')
from keras.layers import ConvLSTM2D, Dense, Flatten
from keras.models import Sequential
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from data import get_image_feature_image_array, get_training_label_array, get_time_delay_training_data
import numpy as np
import math
from dataProcessor import DataProcessor
from conv_lstm import ConvLSTM

samples = None
time = 1
rows = 64
cols = 64
channels = 1
verbose = True
target_image_dimensions = (128,128)

print("Extracting features...")
d = DataProcessor()
root_directory = "../data/cohn_kanade_images"

d.add_feature('hog', {'orientations': 8, 'pixels_per_cell': (16, 16), 'cells_per_block': (1, 1)})
# d.add_feature('lbp', {'n_points': 24, 'radius': 3})

raw_features = d.get_training_data(from_csv=False, dataset_location=root_directory, initial_image_dims=None, target_image_dims=(64, 64), vector=False, time_series=False)
features = np.array([[[feature]] for feature in raw_features])

# labels =
labels = get_training_label_array()
if verbose:
    print('feature shape: ' + str(features.shape))
    print('label shape: ' + str(labels.shape))


print("Creating neural net...")
net = ConvLSTM()

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
net.compile(optimizer="RMSProp", loss="cosine_proximity", metrics=["accuracy"])
net.fit(X_train, y_train, batch_size=10, epochs=100, validation_data=(X_test, y_test),
        callbacks=[ReduceLROnPlateau(), EarlyStopping(patience=3)])

