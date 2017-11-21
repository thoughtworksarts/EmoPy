
class ConvLSTM:
    def __init__(self, verbose=True):
        self.model = self.createModel()

    def createModel(self):
        net = Sequential()
        net.add(ConvLSTM2D(filters=10, kernel_size=(4, 4), activation="sigmoid", input_shape=(time, channels, rows, cols), data_format='channels_first'))
        net.add(Flatten())
        net.add(Dense(units=4, activation="sigmoid"))
        if verbose:
            net.summary()
        return net