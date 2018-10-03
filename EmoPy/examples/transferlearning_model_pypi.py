
from EmoPy.src.csv_data_loader import CSVDataLoader
from EmoPy.src.neuralnets import TransferLearningNN
from EmoPy.src.data_generator import DataGenerator

from pkg_resources import resource_filename

from keras import backend as K
K.set_image_data_format("channels_last")

validation_split = 0.15
verbose = True
model_name = 'inception_v3'

target_dimensions = (128, 128)
raw_dimensions = (48, 48)
fer_dataset_label_map = {'0': 'anger', '2': 'fear'}

print('--------------- Inception-V3 Model -------------------')
print('Loading data...')
csv_file_path = resource_filename('EmoPy.examples','image_data/sample.csv')
data_loader = CSVDataLoader(target_emotion_map=fer_dataset_label_map, datapath=csv_file_path, validation_split=validation_split, image_dimensions=raw_dimensions, csv_label_col=0, csv_image_col=1, out_channels=3)
dataset = data_loader.load_data()

if verbose:
    dataset.print_data_details()

print('Creating training/testing data...')
train_images, train_labels = dataset.get_training_data()
train_gen = DataGenerator().fit(train_images, train_labels)
test_images, test_labels = dataset.get_test_data()
test_gen = DataGenerator().fit(test_images, test_labels)

print('Initializing neural network with InceptionV3 base model...')
model = TransferLearningNN(model_name=model_name, emotion_map=dataset.get_emotion_index_map())

print('Training model...')
model.fit_generator(train_gen.generate(target_dimensions, 10),
                    test_gen.generate(target_dimensions, 10),
                    epochs=10)

# Save model configuration
# model.export_model('output/transfer_learning_model.json','output/transfer_learning_weights.h5',"output/transfer_learning_emotion_map.json", emotion_map)
