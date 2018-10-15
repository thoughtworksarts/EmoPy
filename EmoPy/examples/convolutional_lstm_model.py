from EmoPy.src.data_generator import DataGenerator
from EmoPy.src.directory_data_loader import DirectoryDataLoader
from EmoPy.src.neuralnets import ConvolutionalLstmNN
from pkg_resources import resource_filename

validation_split = 0.15

raw_dimensions = (48, 48)
target_dimensions = (64, 64)
channels = 1
verbose = True

print('--------------- Convolutional LSTM Model -------------------')
print('Loading data...')
directory_path = resource_filename('EmoPy.examples',"image_data/sample_image_series_directory")
data_loader = DirectoryDataLoader(datapath=directory_path, validation_split=validation_split, time_delay=2)
dataset = data_loader.load_data()

if verbose:
    dataset.print_data_details()

print('Preparing training/testing data...')
train_images, train_labels = dataset.get_training_data()
train_gen = DataGenerator(time_delay=dataset.get_time_delay()).fit(train_images, train_labels)
test_images, test_labels = dataset.get_test_data()
test_gen = DataGenerator(time_delay=dataset.get_time_delay()).fit(test_images, test_labels)

print('Training net...')
model = ConvolutionalLstmNN(target_dimensions, channels, dataset.get_emotion_index_map(), time_delay=dataset.get_time_delay())
model.fit_generator(train_gen.generate(target_dimensions, batch_size=5),
                    test_gen.generate(target_dimensions, batch_size=5),
                    epochs=5)

## if you want to save a graph of your model layers.
model.save_model_graph()

# Save model configuration
# model.export_model('output/conv_lstm_model.json','output/conv_lstm_weights.h5',"output/conv_lstm_emotion_map.json", emotion_map)
