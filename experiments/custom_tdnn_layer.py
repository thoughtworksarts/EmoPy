from keras import backend as K
from keras.layers import Dense, Input
from keras.models import Sequential


class TimeDelayLayer(Dense):

    def __init__(self, output_dim, time_delay, **kwargs):
        self.output_dim = output_dim
        self.time_delay = time_delay
        super(TimeDelayLayer, self).__init__(output_dim, **kwargs)


    def build(self, input_shape):
        # input_shape traditionally is (batch_size, steps, input_dim)
        input_dim = input_shape[2:]
        print ('In build method...')
        print ('input_shape: ' + str(input_shape))
        print ('input_dim: ' + str(input_dim))

        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(shape=(self.units, self.time_delay+1)+input_dim,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        super(TimeDelayLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return K.dot(x, self.kernel)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


def main():
    net = Sequential()
    #inputLayer = Input(shape=(64,))
    timeDelayLayer = TimeDelayLayer(4, 2, input_shape=(1,64,64))
    #net.add(inputLayer)
    net.add(timeDelayLayer)

    net.summary()

main()