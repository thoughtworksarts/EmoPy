import sys
sys.path.append('../')
from dataloader import DataLoader
from imageprocessor import ImageProcessor
import numpy as np
from neuralnets import TimeDelayConvNN


print('--------------- Time-Delay Convolutional Model -------------------')
print('Loading data...')
directory_path = "image_data/sample_image_series_directory"
target_dimensions = (64, 64)
time_delay = 3
channels = 1
verbose = True

dataLoader = DataLoader(from_csv=False, datapath=directory_path, time_steps=3)
image_data, labels, label_map = dataLoader.get_data()
if verbose:
    print('raw image data shape: ' + str(image_data.shape))
label_count = len(labels[0])

print('Processing data...')
imageProcessor = ImageProcessor(image_data, target_dimensions=target_dimensions, time_series=True)
image_array = imageProcessor.process_training_data()
image_data = np.array([[image] for image in image_array])
if verbose:
    print ('processed image data shape: ' + str(image_data.shape))

print('Creating training/testing data...')
validation_split = 0.15


print('Training net...')
model = TimeDelayConvNN(target_dimensions, time_delay, channels, label_count)
model.fit(image_data, labels, validation_split)