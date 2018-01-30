import sys

sys.path.append('../')
from data_generator import DataGenerator
from imageprocessor import ImageProcessor
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
label_count = len(labels[0])

print('Processing data...')
# <<<<<<< HEAD
imageProcessor = ImageProcessor(image_data, target_dimensions=target_dimensions)
image_array = imageProcessor.process_training_data()
# image_data = np.array([[image] for image in image_array])
# =======
# imageProcessor = ImageProcessor(image_data, target_dimensions=target_dimensions, rgb=False, channels=1)
image_data = imageProcessor.process_training_data()
# >>>>>>> 3692109a392c5f90829bae24133b58d113b0eab2
if verbose:
    print('processed image data shape: ' + str(image_data.shape))

print('Creating training/testing data...')
validation_split = 0.15
X_train, X_test, y_train, y_test = train_test_split(image_data, labels,
                                                    test_size=validation_split, random_state=42, stratify=labels)


train_gen = DataGenerator().fit(X_train, y_train)
test_gen = DataGenerator().fit(X_test, y_test)
print('Training net...')
model = ConvolutionalNN(target_dimensions, channels, label_count)
model.fit_generator(train_gen.generate(batch_size=10), test_gen.generate(batch_size=10), epochs=10)
