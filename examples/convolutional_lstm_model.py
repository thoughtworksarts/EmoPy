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
imageProcessor = ImageProcessor(images, target_dimensions=target_dimensions, rgb=False, channels=1)
images = imageProcessor.process_training_data()
if verbose:
	print ('processed image shape: ' + str(images.shape))

print('Extracting features...')
featureExtractor = FeatureExtractor(images, return_2d_array=True)
featureExtractor.add_feature('hog', {'orientations': 8, 'pixels_per_cell': (16, 16), 'cells_per_block': (1, 1)})
raw_features = featureExtractor.extract()
features = list()
for feature in raw_features:
   features.append([[feature]])
features = np.array(features)
if verbose:
    print('feature shape: ' + str(features.shape))
    print('label shape: ' + str(labels.shape))

print('Creating training/testing data...')
validation_split = 0.15

print('Training net...')
model = ConvolutionalLstmNN(target_dimensions, channels, target_labels, time_delay=time_delay)
model.fit(features, labels, validation_split)