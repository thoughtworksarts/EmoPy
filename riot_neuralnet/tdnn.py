from skimage import data, color, exposure, io
from skimage.feature import hog
import matplotlib.pyplot as plt
import keras
import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv3D
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, Callback, ReduceLROnPlateau, LearningRateScheduler, EarlyStopping, TensorBoard
from keras.callbacks import LambdaCallback
from keras import backend as K

import tensorflow as tf

def extractFeatureVector(imageFile):
    image = io.imread(imageFile)
    image.resize((64,64))
    image = color.rgb2gray(image)
    featureVector, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualise=True) #, transform_sqrt=True, feature_vector=False)
    return hog_image

def createCNN(verbose=False):
    model = Sequential()
    model.add(Conv3D(filters = 10, kernel_size=(1,64,64), activation='sigmoid', input_shape=(1,1,64,64), padding='same'))
    # model.add(Conv2D(filters=12, kernel_size=(3,3),
    # activation='relu', input_shape=(28,28,1), padding='same'))
    model.add(Flatten())
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(optimizer='sgd',
                  loss='mean_squared_error',
                  metrics=['accuracy'])
    if verbose:
        model.summary()

    return model


def train(model):
    x_temp = list()
    x_temp.append([[extractFeatureVector('images/S502_001_00000001.png')]])
    x_temp.append([[extractFeatureVector('images/S502_001_00000002.png')]])
    x_temp.append([[extractFeatureVector('images/S502_001_00000003.png')]])
    x_temp.append([[extractFeatureVector('images/S502_001_00000004.png')]])
    x_temp.append([[extractFeatureVector('images/S502_001_00000005.png')]])
    x_temp.append([[extractFeatureVector('images/S502_001_00000006.png')]])
    x_temp.append([[extractFeatureVector('images/S502_001_00000007.png')]])
    x_temp.append([[extractFeatureVector('images/S502_001_00000008.png')]])
    x_temp.append([[extractFeatureVector('images/S502_001_00000009.png')]])
    x_temp.append([[extractFeatureVector('images/S502_001_00000010.png')]])

    X_train = np.array(x_temp)
    print (X_train.shape)
    y_train = np.array([random.uniform(0.8,1)]*10)
    X_test = np.array([[[extractFeatureVector('images/S502_001_00000001.png')]]])
    print(X_test.shape)
    y_test = np.array([1])

    history = model.fit(X_train,
                        y_train,
                        batch_size=10,
                        epochs=5,
                        validation_data=(X_test, y_test),
                        callbacks=[ReduceLROnPlateau(), EarlyStopping(patience=3)]
                        )

def main():
    model = createCNN(verbose=True)
    train(model)

main()