import matplotlib.pyplot as plt
import pdb

from skimage.feature import hog
from skimage import data, color, exposure, io

from sknn.mlp import Regressor, Convolution, Layer, Native
from lasagne.layers import Conv3DLayer, Conv2DLayer, DenseLayer, InputLayer
from lasagne import nonlinearities as nl

import numpy as np



def extractFeatureVector(imageFile, verbose=False):
    image = io.imread(imageFile)
    image.resize((400,400))
    image = color.rgb2gray(image)
    print 'image shape: ' + str(image.shape)
    featureVector, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualise=True) #, transform_sqrt=True, feature_vector=False)

    if (verbose):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

        ax1.axis('off')
        ax1.imshow(image, cmap=plt.cm.gray)
        ax1.set_title('Input image')
        ax1.set_adjustable('box-forced')

        # Rescale histogram for better display
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))

        ax2.axis('off')
        ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
        ax2.set_title('Histogram of Oriented Gradients')
        ax1.set_adjustable('box-forced')
        plt.show()

    return hog_image


def getNeuralNet():
    incoming = (1, 1, 2, 400, 400)  # batch size, input channels, input rows, input cols, input.. time
    node_count = 10
    inputLayer = Native(InputLayer, (2, 400, 400))
    hiddenLayer = Native(Conv3DLayer, incoming, node_count, (2, 400, 400), pad=0, untie_biases=True)
    hiddenLayer2 = Native(Conv2DLayer, (None, 1, 400, 400), 16, 3)
    # lasagne.layers.Conv3DLayer(incoming, num_filters, filter_size, stride=(1, 1, 1), pad=0, untie_biases=False, W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.), nonlinearity=lasagne.nonlinearities.rectify, flip_filters=True, convolution=theano.tensor.nnet.conv3d, **kwargs)
    outputLayer = Layer('Linear')
    net = Regressor(layers=[hiddenLayer, outputLayer], learning_rate=0.01, n_iter=20)
    return net


def train(net, x_train=None, y_train=None):
    x_train = np.array([[[extractFeatureVector('images/S502_001_00000001.png'), extractFeatureVector('images/S502_001_00000002.png')]]])
    # temp.append(extractFeatureVector('images/S502_001_00000001.png'))
    # temp.append(extractFeatureVector('images/S502_001_00000002.png'))
    # x_train = np.array(temp)
    print 'x_train shape: ' + str(x_train.shape)
    y_train = np.array([1])
    y_train = np.reshape(y_train, y_train.shape + (1,))
    print 'y_train shape: ' + str(y_train.shape)

    pdb.set_trace()

    net.fit(x_train, y_train)

def main():
    imageFile = '/Users/aperez/Documents/TW/RIOT/Riot_python/images/S502_001_00000010.png'
    featureVector = np.array([extractFeatureVector(imageFile, verbose=False)])
    net = getNeuralNet()
    train(net)
    #print "Prediction: " + str(net.predict(featureVector))



main()