import sys
sys.path.append('../feature')
sys.path.append('../data')
from dataProcessor import DataProcessor
from transferModel import TransferModel



print('Creating NN with InceptionV3 base model...')
model = TransferModel(model_name='inception_v3')

print('Extracting training data...')

target_image_dims = (128,128)

d = DataProcessor()
root_directory = "../data/cohn_kanade_images"
csv_file_path = "../data/fer2013/fer2013.csv"

d.add_feature('hog', {'orientations': 8, 'pixels_per_cell': (4, 4), 'cells_per_block': (1, 1)})

X_train, y_train, X_test, y_test = d.get_training_data(from_csv=True, dataset_location=csv_file_path, target_image_dims=target_image_dims, initial_image_dims=(48, 48), label_index=0, image_index=1, vector=False, time_series=False)

print('X_train shape: ' + str(X_train.shape))
print('y_train shape: ' + str(y_train.shape))
print('X_test shape: ' + str(X_test.shape))
print('y_test shape: ' + str(y_test.shape))


print ('Training model...')
print('numLayers: ' + str(len(model.model.layers)))
model.fit(X_train, y_train, X_test, y_test)

trained_model_output_filepath = '../trained_models/inception_v3_model_1.h5'
model.model.save(trained_model_output_filepath)


