import sys
sys.path.append('../data')
sys.path.append('../fer')
from imageprocessor import ImageProcessor
from neuralnets import TransferLearningNN, TimeDelayNN, ConvolutionalLstmNN
from featureextractor import FeatureExtractor

verbose = True
target_dimensions = (128, 128)
raw_dimensions = (48, 48)
target_labels = [0,1,2,3]
print('--------------- Regression + TimeDelayNN Model -------------------')
print('Collecting data...')
root_directory = '../data/cohn_kanade_images'
imageProcessor = ImageProcessor(from_csv=False, target_labels=target_labels, datapath=root_directory, target_dimensions=target_dimensions, raw_dimensions=None)
images, labels = imageProcessor.get_training_data()

print ('images shape: ' + str(images.shape))
print ('labels shape: ' + str(labels.shape))
print('Extracting features...')
featureExtractor = FeatureExtractor(images, return_2d_array=False)
featureExtractor.add_feature('hog', {'orientations': 8, 'pixels_per_cell': (16, 16), 'cells_per_block': (1, 1)})
# featureExtractor.add_feature('lbp', {'n_points': 24, 'radius': 3})
features = featureExtractor.extract()
print ("features shape: " + str(features.shape))

tdnn = TimeDelayNN(len(features[0]), verbose=True)
tdnn.fit(features, labels, 0.15)