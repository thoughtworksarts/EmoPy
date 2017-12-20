import sys
sys.path.append('../')
from imageprocessor import ImageProcessor
from neuralnets import TransferLearningNN
from skimage import color, io
import numpy as np
from matplotlib import pyplot as plt

verbose = True
target_dimensions = (128, 128)
raw_dimensions = (48, 48)
target_labels = [0,1,2,3,4,5,6]
model_name = 'inception_v3'

print('Extracting training data...')
csv_file_path = "image_data/sample.csv"
imageProcessor = ImageProcessor(from_csv=True, target_labels=target_labels, datapath=csv_file_path, target_dimensions=target_dimensions, raw_dimensions=raw_dimensions, csv_label_col=0, csv_image_col=1)

features, labels = imageProcessor.get_training_data()

print('--------------- Inception-V3 Model -------------------')
print('Initializing neural network with InceptionV3 base model...')
model = TransferLearningNN(model_name=model_name, target_labels=target_labels)

print('Training model...')
print('numLayers: ' + str(len(model.model.layers)))
model.fit(features, labels, 0.15)
