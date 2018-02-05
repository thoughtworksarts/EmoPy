import sys
sys.path.append('../')
from imageprocessor import ImageProcessor
from dataloader import DataLoader
from neuralnets import ConvolutionalLstmNN
from featureextractor import FeatureExtractor
import numpy as np

time_delay = 1
raw_dimensions = (48, 48)
target_dimensions = (64, 64)
channels = 1
verbose = True
using_feature_extraction = True
target_labels = [0,1,2,3,4,5,6]


print('--------------- Convolutional LSTM Model -------------------')
print('Loading data...')
csv_file_path = "image_data/sample.csv"

dataLoader = DataLoader(from_csv=True, target_labels=target_labels, datapath=csv_file_path, image_dimensions=raw_dimensions, csv_label_col=0, csv_image_col=1)
images, labels = dataLoader.get_data()
if verbose:
    print('raw image shape: ' + str(images.shape))

print('Processing data...')
imageProcessor = ImageProcessor(images, target_dimensions=target_dimensions)
images = imageProcessor.process_training_data()
if verbose:
	print ('processed image shape: ' + str(images.shape))
features = list()
features = np.array([[np.array([image]).reshape(list(target_dimensions)+[channels])] for image in images])
if verbose:
    print('feature shape: ' + str(features.shape))
    print('label shape: ' + str(labels.shape))

print('Training net...')
validation_split = 0.15
model = ConvolutionalLstmNN(target_dimensions, channels, target_labels, time_delay=time_delay)
model.fit(features, labels, validation_split)

## if you want to save a graph of your model layers.
model.save_model_graph()