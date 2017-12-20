import sys
sys.path.append('../')
from imageprocessor import ImageProcessor
from neuralnets import TimeDelayNN
from featureextractor import FeatureExtractor


# ------------- PARAMETERS -------------- #
verbose = True
target_dimensions = (128, 128)
raw_dimensions = (48, 48)
target_labels = [0,1,2,3]

print('--------------- Regression + TimeDelayNN Model -------------------')
print('Collecting data...')
root_directory = 'image_data/sample_directory'
imageProcessor = ImageProcessor(from_csv=False, target_labels=target_labels, datapath=root_directory, target_dimensions=target_dimensions, raw_dimensions=None)
images, labels = imageProcessor.get_training_data()

print ('images shape: ' + str(images.shape))
print ('labels shape: ' + str(labels.shape))
print('Extracting features...')
featureExtractor = FeatureExtractor(images, return_2d_array=False)
featureExtractor.add_feature('hog', {'orientations': 8, 'pixels_per_cell': (16, 16), 'cells_per_block': (1, 1)})
features = featureExtractor.extract()
print ("features shape: " + str(features.shape))

tdnn = TimeDelayNN(len(features[0]), verbose=True)
tdnn.fit(features, labels, 0.15)