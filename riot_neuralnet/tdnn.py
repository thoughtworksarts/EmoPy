from keras.layers import Conv3D
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

class TDNN:

    def __init__(self, verbose=False):
        net = Sequential()
        net.add(Conv3D(filters = 10, kernel_size=(1,64,64), activation='sigmoid', input_shape=(1,1,64,64), padding='same'))
        net.add(Flatten())
        net.add(Dense(units=4, activation='sigmoid'))
        net.compile(optimizer='sgd',
                      loss='mean_squared_error',
                      metrics=['accuracy'])
        if verbose:
            net.summary()

        self.net = net

    def train(self, X_train, y_train, X_test, y_test):
        self.net.fit(X_train,
                            y_train,
                            batch_size=10,
                            epochs=5,
                            validation_data=(X_test, y_test),
                            callbacks=[ReduceLROnPlateau(), EarlyStopping(patience=3)]
                            )

