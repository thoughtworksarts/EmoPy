import matplotlib.pyplot as plt

from skimage.feature import hog
from skimage import data, color, exposure, io

from sknn.mlp import Regressor, Convolution, Layer

import numpy as np



def extractFeatureVector(imageFile, verbose=False):
    image = color.rgb2gray(io.imread(imageFile))
    featureVector, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualise=True)

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

    return featureVector


def getNeuralNet():
    hiddenLayer = Convolution('Sigmoid', channels=10, kernel_shape=(1,10800), kernel_stride=(1,10800))
    #outputLayer = Layer('Softmax')
    outputLayer = Convolution('Sigmoid', channels=1, kernel_shape=(1,10800), kernel_stride=(1,10800))
    net = Regressor(layers=[hiddenLayer, outputLayer], learning_rate=0.01, n_iter=20)
    return net


def train(net, x_train=None, y_train=None):
    x_train = np.ndarray(shape=(1,5*10800))
    np.append(x_train, extractFeatureVector('/Users/aperez/Documents/TW/RIOT/Riot_python/images/S502_001_00000001.png'))
    np.append(x_train, extractFeatureVector('/Users/aperez/Documents/TW/RIOT/Riot_python/images/S502_001_00000002.png'))
    np.append(x_train, extractFeatureVector('/Users/aperez/Documents/TW/RIOT/Riot_python/images/S502_001_00000003.png'))
    np.append(x_train, extractFeatureVector('/Users/aperez/Documents/TW/RIOT/Riot_python/images/S502_001_00000004.png'))
    np.append(x_train, extractFeatureVector('/Users/aperez/Documents/TW/RIOT/Riot_python/images/S502_001_00000005.png'))
    print "x_train: " + str(x_train.shape)
    y_train = np.ndarray(shape=(1,5))
    np.append(y_train, 1)
    np.append(y_train, 1)
    np.append(y_train, 1)
    np.append(y_train, 1)
    np.append(y_train, 1)
    print "y_train: " + str(y_train.shape)
    net.fit(x_train, y_train)

def main():
    imageFile = '/Users/aperez/Documents/TW/RIOT/Riot_python/images/S502_001_00000001.png'
    featureVector = extractFeatureVector(imageFile, verbose=False)
    net = getNeuralNet()
    train(net)
    #print('%s' % len(featureVector))



main()