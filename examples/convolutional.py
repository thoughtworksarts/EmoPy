import sys
sys.path.append('../')
from imageprocessor import ImageProcessor
from dataloader import DataLoader
from neuralnets import ConvolutionalNN
import numpy as np

target_dimensions = (64, 64)
channels = 1
verbose = True

print('--------------- Convolutional Model -------------------')
print('Loading data...')
directory_path = "image_data/sample_image_directory"

dataLoader = DataLoader(from_csv=False, datapath=directory_path)
image_data, labels = dataLoader.get_data()
if verbose:
    print('raw image data shape: ' + str(image_data.shape))
label_count = len(labels[0])

print('Processing data...')
imageProcessor = ImageProcessor(image_data, target_dimensions=target_dimensions, rgb=False, channels=1)
image_array = imageProcessor.process_training_data()
image_data = np.array([[image] for image in image_array])
if verbose:
	print ('processed image data shape: ' + str(image_data.shape))

print('Creating training/testing data...')
validation_split = 0.15

print('Training net...')
model = ConvolutionalNN(target_dimensions, channels, label_count)
model.fit(image_data, labels, validation_split)