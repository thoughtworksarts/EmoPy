import sys
sys.path.append('../data')
sys.path.append('../fer')
from imageprocessor import ImageProcessor
from neuralnets import TransferLearningNN, TimeDelayNN, ConvolutionalLstmNN


verbose = True
target_dimensions = (128, 128)
raw_dimensions = (48, 48)
target_labels = [0,1,2,3,4,5,6]
model_name = 'resnet50'

print('--------------- Inception-V3 Model -------------------')
print('Creating NN with InceptionV3 base model...')
model = TransferLearningNN(model_name=model_name, target_labels=target_labels)

print('Extracting training data...')


csv_file_path = "../data/fer2013/fer2013.csv"
imageProcessor = ImageProcessor(from_csv=True, target_labels=target_labels, datapath=csv_file_path, target_dimensions=target_dimensions, raw_dimensions=raw_dimensions, csv_label_col=0, csv_image_col=1)

features, labels = imageProcessor.get_training_data()

print ('Training model...')
print('numLayers: ' + str(len(model.model.layers)))
model.fit(features, labels, 0.15)

# -- Export model and trained weights to json and h5 files
# trained_weights_output_filepath = '../trained_models/inception_v3_weights.h5'
# model.model.save_weights(trained_weights_output_filepath)
# trained_model_output_filepath = '../trained_models/inception_v3_model.json'
# model_json_string = model.model.to_json()
# model_json_file = open(trained_model_output_filepath, 'w')
# model_json_file.write(model_json_string)
# model_json_file.close()

