import sys
sys.path.append('../')
from dataloader import DataLoader
from imageprocessor import ImageProcessor
from neuralnets import TransferLearningNN

verbose = True
target_dimensions = (128, 128)
raw_dimensions = (48, 48)
target_labels = [0,1,2,3,4,5,6]
model_name = 'inception_v3'

print('Loading data...')
csv_file_path = "image_data/sample.csv"

dataLoader = DataLoader(from_csv=True, target_labels=target_labels, datapath=csv_file_path, image_dimensions=raw_dimensions, csv_label_col=0, csv_image_col=1)
images, labels = dataLoader.get_data()
if verbose:
    print('raw image shape: ' + str(images.shape))

print('Processing data...')
imageProcessor = ImageProcessor(images, target_dimensions=target_dimensions, rgb=True)
images = imageProcessor.process_training_data()
if verbose:
    print ('processed image shape: ' + str(images.shape))

print('--------------- Inception-V3 Model -------------------')
print('Initializing neural network with InceptionV3 base model...')
model = TransferLearningNN(model_name=model_name, target_labels=target_labels)

print('Training model...')
print('numLayers: ' + str(len(model.model.layers)))
model.fit(images, labels, 0.15)
