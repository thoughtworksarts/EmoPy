import numpy as np
import random
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.layers import Conv3D
from keras.layers import Dense, Flatten
from keras.models import Sequential
from feature import Feature


def createCNN(verbose=False):
    model = Sequential()
    model.add(Conv3D(filters = 10, kernel_size=(1,64,64), activation='sigmoid', input_shape=(1,1,64,64), padding='same'))
    model.add(Flatten())
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(optimizer='sgd',
                  loss='mean_squared_error',
                  metrics=['accuracy'])
    if verbose:
        model.summary()

    return model


def train(model):
    feature = Feature()
    x_temp = list()
    x_temp.append([[feature.extractFeatureVector('../images/S502_001_00000001.png')]])
    x_temp.append([[feature.extractFeatureVector('../images/S502_001_00000002.png')]])
    x_temp.append([[feature.extractFeatureVector('../images/S502_001_00000003.png')]])
    x_temp.append([[feature.extractFeatureVector('../images/S502_001_00000004.png')]])
    x_temp.append([[feature.extractFeatureVector('../images/S502_001_00000005.png')]])
    x_temp.append([[feature.extractFeatureVector('../images/S502_001_00000006.png')]])
    x_temp.append([[feature.extractFeatureVector('../images/S502_001_00000007.png')]])
    x_temp.append([[feature.extractFeatureVector('../images/S502_001_00000008.png')]])
    x_temp.append([[feature.extractFeatureVector('../images/S502_001_00000009.png')]])
    x_temp.append([[feature.extractFeatureVector('../images/S502_001_00000010.png')]])

    X_train = np.array(x_temp)
    print (X_train.shape)
    y_train = np.array([random.uniform(0.8,1)]*10)
    X_test = np.array([[[feature.extractFeatureVector('../images/S502_001_00000001.png')]]])
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