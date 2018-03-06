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
model_name = 'inception_v3'
fer_dataset_label_map = {'0': 'anger', '2': 'fear'}

print('Loading data...')
csv_file_path = "image_data/sample.csv"

data_loader = DataLoader(from_csv=True, emotion_map=fer_dataset_label_map, datapath=csv_file_path,
                         image_dimensions=raw_dimensions, csv_label_col=0, csv_image_col=1, out_channels=3)
images, labels, emotion_map = data_loader.get_data()
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
model = TransferLearningNN(model_name=model_name, emotion_map=emotion_map)

print('Training model...')
print('numLayers: ' + str(len(model.model.layers)))

model.fit_generator(train_gen.generate(target_dimensions, 10),
                    test_gen.generate(target_dimensions, 10),
                    epochs=10)

# Save model configuration
# model.export_model('output/transfer_learning_model.json','output/transfer_learning_weights.h5',"output/transfer_learning_emotion_map.json", emotion_map)