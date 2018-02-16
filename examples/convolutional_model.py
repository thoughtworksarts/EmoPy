import sys

sys.path.append('../')
from data_generator import DataGenerator
from dataloader import DataLoader
from neuralnets import ConvolutionalNN
from sklearn.model_selection import train_test_split

target_dimensions = (64, 64)
channels = 1
verbose = True

print('--------------- Convolutional Model -------------------')
print('Loading data...')
directory_path = "image_data/sample_image_directory"

dataLoader = DataLoader(from_csv=False, datapath=directory_path)
image_data, labels, label_map = dataLoader.get_data()
if verbose:
    print('raw image data shape: ' + str(image_data.shape))
label_count = labels.shape[1]

print('Creating training/testing data...')
validation_split = 0.15
X_train, X_test, y_train, y_test = train_test_split(image_data, labels,
                                                    test_size=validation_split, random_state=42, stratify=labels)
train_gen = DataGenerator().fit(X_train, y_train)
test_gen = DataGenerator().fit(X_test, y_test)
print('Training net...')
model = ConvolutionalNN(target_dimensions, channels, label_count)
# model.fit(image_data, labels, validation_split)
model.fit_generator(train_gen.generate(target_dimensions, batch_size=5),
                    test_gen.generate(target_dimensions, batch_size=5),
                    epochs=10)
