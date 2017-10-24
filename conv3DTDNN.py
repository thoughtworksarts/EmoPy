import matplotlib.pyplot as plt
import pdb
import time

from skimage.feature import hog
from skimage import data, color, exposure, io

from lasagne.layers import Conv3DLayer, Conv2DLayer, DenseLayer, InputLayer
from lasagne import nonlinearities as nl

import numpy as np

import lasagne
import theano
import theano.tensor as T



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

def train():
    X_train = np.array([[[extractFeatureVector('images/S502_001_00000001.png'), extractFeatureVector('images/S502_001_00000002.png')]]])
    print 'x_train shape: ' + str(X_train.shape)
    y_train = np.array([1])
    y_train = np.reshape(y_train, y_train.shape + (1,))
    print 'y_train shape: ' + str(y_train.shape)

    input_var = T.tensor5('inputs')
    target_var = T.matrix('targets')

    node_count = 10
    inputLayer = InputLayer((1, 1, 2, 400, 400), input_var=input_var)
    hiddenLayer = Conv3DLayer(inputLayer, node_count, (2, 400, 400), pad=0, untie_biases=True)
    net = DenseLayer(hiddenLayer, num_units=1, nonlinearity=nl.softmax)

    predictor = lasagne.layers.get_output(net)
    loss = lasagne.objectives.squared_error(predictor, target_var)
    loss = loss.mean()

    params = lasagne.layers.get_all_params(net, trainable=True)
    updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.01, momentum=0.9)

    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    # Collect points for final plot
    train_err_plot = []

    # Finally, launch the training loop.
    print "Starting training..."

    # We iterate over epochs:
    num_epochs = 100
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        start_time = time.time()
        train_err = train_fn(X_train, y_train)

        # Then we print the results for this epoch:
        print "Epoch %s of %s took %.3fs" % (epoch+1, num_epochs, time.time()-start_time)
        print "  training loss:\t\t%s" % train_err

        # Save accuracy to show later
        train_err_plot.append(train_err)

    # Show plot
    plt.plot(train_err_plot)
    plt.title('Graph')
    plt.xlabel('Epochs')
    plt.ylabel('Training loss')
    plt.tight_layout()
    plt.show()


def main():
    train()



main()