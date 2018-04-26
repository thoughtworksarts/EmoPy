import sys
sys.path.append('../')
from src.data_generator import DataGenerator
from src.dataloader import DataLoader
from src.neuralnets import ConvolutionalLstmNN

validation_split = 0.15

raw_dimensions = (48, 48)
target_dimensions = (64, 64)
channels = 1
verbose = True

print('--------------- Convolutional LSTM Model -------------------')
print('Loading data...')
directory_path = "image_data/sample_image_series_directory"
data_loader = DataLoader(from_csv=False, datapath=directory_path, validation_split=validation_split, time_delay=2)
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
                    epochs=10)

## if you want to save a graph of your model layers.
model.save_model_graph()

# Save model configuration
# model.export_model('output/conv_lstm_model.json','output/conv_lstm_weights.h5',"output/conv_lstm_emotion_map.json", emotion_map)