import sys
sys.path.append('../')

from sklearn.model_selection import train_test_split

from src.data_generator import DataGenerator
from src.dataloader import DataLoader
from src.neuralnets import TimeDelayConvNN

print('--------------- Time-Delay Convolutional Model -------------------')
print('Loading data...')
directory_path = "image_data/sample_image_series_directory"
target_dimensions = (64, 64)
channels = 1
verbose = True

data_loader = DataLoader(from_csv=False, datapath=directory_path, time_delay=3)
dataset = data_loader.get_data()
labels = dataset.get_labels()
images = dataset.get_images()

if verbose:
    print('raw image data shape: ' + str(images.shape))
label_count = len(labels[0])

print('Creating training/testing data...')
validation_split = 0.25
X_train, X_test, y_train, y_test = train_test_split(images, labels,
                                                    test_size=validation_split, random_state=42, stratify=labels)
train_gen = DataGenerator(time_delay=dataset.get_time_delay()).fit(X_train, y_train)
test_gen = DataGenerator(time_delay=dataset.get_time_delay()).fit(X_test, y_test)

print('Training net...')
model = TimeDelayConvNN(target_dimensions, channels, emotion_map=dataset.get_emotion_index_map(), time_delay=dataset.get_time_delay())
model.fit_generator(train_gen.generate(target_dimensions, batch_size=10),
                    test_gen.generate(target_dimensions, batch_size=10),
                    epochs=10)

# Save model configuration
# model.export_model('output/time_delay_model.json','output/time_delay_weights.h5',"output/time_delay_emotion_map.json", emotion_map)