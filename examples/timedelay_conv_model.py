import sys
sys.path.append('../')
from src.data_generator import DataGenerator
from src.dataloader import DataLoader
from src.neuralnets import TimeDelayConvNN

validation_split = 0.25

target_dimensions = (64, 64)
channels = 1
verbose = True

print('--------------- Time-Delay Convolutional Model -------------------')
print('Loading data...')
directory_path = "image_data/sample_image_series_directory"
data_loader = DataLoader(from_csv=False, datapath=directory_path, validation_split=validation_split, time_delay=3)
dataset = data_loader.load_data()

if verbose:
    dataset.print_data_details()

print('Preparing training/testing data...')
train_images, train_labels = dataset.get_training_data()
train_gen = DataGenerator(time_delay=dataset.get_time_delay()).fit(train_images, train_labels)
test_images, test_labels = dataset.get_test_data()
test_gen = DataGenerator(time_delay=dataset.get_time_delay()).fit(test_images, test_labels)

print('Training net...')
model = TimeDelayConvNN(target_dimensions, channels, emotion_map=dataset.get_emotion_index_map(), time_delay=dataset.get_time_delay())
model.fit_generator(train_gen.generate(target_dimensions, batch_size=10),
                    test_gen.generate(target_dimensions, batch_size=10),
                    epochs=10)

# Save model configuration
# model.export_model('output/time_delay_model.json','output/time_delay_weights.h5',"output/time_delay_emotion_map.json", emotion_map)