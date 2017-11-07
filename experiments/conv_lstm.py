import sys
sys.path.append('../feature')
from keras.layers import ConvLSTM2D, Dense, Flatten
from keras.models import Sequential
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from data import get_image_feature_image_array, get_training_label_array, get_time_delay_training_data
import numpy as np
import math

samples = None
time = 1
rows = 64
cols = 64
channels = 1
verbose = True

print("Extracting features...")
features = get_image_feature_image_array()
labels = get_training_label_array()
if verbose:
    print('feature shape: ' + str(features.shape))
    print('label shape: ' + str(labels.shape))


print("Creating neural net...")
net = Sequential()
net.add(ConvLSTM2D(filters=10, kernel_size=(4, 4), activation="sigmoid", input_shape=(time, channels, rows, cols), data_format='channels_first'))
net.add(Flatten())
net.add(Dense(units=4, activation="sigmoid"))
if verbose:
    net.summary()


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

