import sys
sys.path.append('../')
from dataloader import DataLoader
from imageprocessor import ImageProcessor
from neuralnets import TimeDelayNN
from featureextractor import FeatureExtractor


verbose = True
target_dimensions = (128, 128)
raw_dimensions = (48, 48)
target_labels = [0,1,2,3]

print('--------------- Regression + TimeDelayNN Model -------------------')
print('Loading data...')
root_directory = 'image_data/sample_image_series_directory'

dataLoader = DataLoader(from_csv=False, target_labels=target_labels, datapath=root_directory, image_dimensions=raw_dimensions, avep=True)
images, labels = dataLoader.get_data()
if verbose:
    print('raw image data shape: ' + str(images.shape))

print('Processing data...')
imageProcessor = ImageProcessor(images, target_dimensions=target_dimensions, channels=1)
images = imageProcessor.process_training_data()
if verbose:
    print ('processed image data shape: ' + str(images.shape))
    print ('label data shape: ' + str(labels.shape))

print('Extracting features...')
featureExtractor = FeatureExtractor(images, return_2d_array=False)
featureExtractor.add_feature('hog', {'orientations': 8, 'pixels_per_cell': (16, 16), 'cells_per_block': (1, 1)})
features = featureExtractor.extract()
print ("features shape: " + str(features.shape))

tdnn = TimeDelayNN(len(features[0]), verbose=True)
tdnn.fit(features, labels, 0.15)