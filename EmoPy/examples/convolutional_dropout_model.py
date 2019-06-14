from EmoPy.src.fermodel import FERModel
from EmoPy.src.directory_data_loader import DirectoryDataLoader
from EmoPy.src.csv_data_loader import CSVDataLoader
from EmoPy.src.data_generator import DataGenerator
from EmoPy.src.neuralnets import ConvolutionalNNDropout
from sklearn.model_selection import train_test_split
import numpy as np

from pkg_resources import resource_filename,resource_exists

validation_split = 0.15

target_dimensions = (48, 48)
channels = 1
verbose = True

#fer_dataset_label_map = {'0': 'anger', '1' : 'disgust', '2': 'fear', '3' : 'happiness', '4' : 'sadness', '5' : 'surprise', '6' : 'calm'}
fer_dataset_label_map = {'0': 'anger', '1' : 'disgust', '2': 'fear', '3' : 'happiness'}

print('--------------- Convolutional Dropout Model -------------------')
print('Loading data...')
csv_file_path = resource_filename('EmoPy.examples','image_data/fer2013.csv')
data_loader = CSVDataLoader(target_emotion_map=fer_dataset_label_map, datapath=csv_file_path, validation_split=validation_split, 
	image_dimensions=target_dimensions, csv_label_col=0, csv_image_col=1, out_channels=1)
dataset = data_loader.load_data()


if verbose:
    dataset.print_data_details()

print('Preparing training/testing data...')
train_images, train_labels = dataset.get_training_data()
train_gen = DataGenerator().fit(train_images, train_labels)
test_images, test_labels = dataset.get_test_data()
test_gen = DataGenerator().fit(test_images, test_labels)

X_train, X_valid, y_train, y_valid = train_test_split(train_images, train_labels, test_size=0.1, random_state=41)

print('Training net...')
model = ConvolutionalNNDropout(target_dimensions, channels, dataset.get_emotion_index_map(), verbose=True)
model.fit_generator(train_gen.generate(target_dimensions, batch_size=5),
                    test_gen.generate(target_dimensions, batch_size=5),
                    epochs=15)
#model.fit_generator(train_gen.generate(target_dimensions, batch_size=5),validation_data=(np.array(X_valid), np.array(y_valid)),
                    #epochs=100)

# Save model configuration
# model.export_model('output/conv2d_model.json','output/conv2d_weights.h5',"output/conv2d_emotion_map.json", emotion_map)
