import sys
from sklearn.model_selection import train_test_split

sys.path.append('../')
from src.dataloader import DataLoader
from src.neuralnets import TransferLearningNN
from src.data_generator import DataGenerator
from keras import backend as K
K.set_image_data_format("channels_last")

verbose = True
target_dimensions = (128, 128)
raw_dimensions = (48, 48)
target_labels = [0, 1, 2, 3, 4, 5, 6]
model_name = 'inception_v3'

print('Loading data...')
csv_file_path = "image_data/sample.csv"

dataLoader = DataLoader(from_csv=True, target_labels=target_labels, datapath=csv_file_path,
                        image_dimensions=raw_dimensions, csv_label_col=0, csv_image_col=1, out_channels=3)
images, labels = dataLoader.get_data()
if verbose:
    print('raw image shape: ' + str(images.shape))

print('Creating training/testing data...')
validation_split = 0.15
X_train, X_test, y_train, y_test = train_test_split(images, labels,
                                                    test_size=validation_split, random_state=42, stratify=labels)
train_gen = DataGenerator().fit(X_train, y_train)
test_gen = DataGenerator().fit(X_test, y_test)

print('--------------- Inception-V3 Model -------------------')
print('Initializing neural network with InceptionV3 base model...')
model = TransferLearningNN(model_name=model_name, target_labels=target_labels)

print('Training model...')
print('numLayers: ' + str(len(model.model.layers)))

model.fit_generator(train_gen.generate(target_dimensions, 10),
                    test_gen.generate(target_dimensions, 10),
                    epochs=10)
