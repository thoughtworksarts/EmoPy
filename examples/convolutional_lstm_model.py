import sys
sys.path.append('../')

from src.data_generator import DataGenerator
from src.dataloader import DataLoader
from src.neuralnets import ConvolutionalLstmNN
from sklearn.model_selection import train_test_split

time_delay = 2
raw_dimensions = (48, 48)
target_dimensions = (64, 64)
channels = 1
verbose = True
using_feature_extraction = True
target_labels = [0,1,2,3,4,5]


print('--------------- Convolutional LSTM Model -------------------')
print('Loading data...')
directory_path = "image_data/sample_image_series_directory"
dataLoader = DataLoader(from_csv=False, datapath=directory_path, time_steps=time_delay)
image_data, labels, label_map = dataLoader.get_data()
if verbose:
    print('raw image data shape: ' + str(image_data.shape))
label_count = len(labels[0])

print('Training net...')
validation_split = 0.15
X_train, X_test, y_train, y_test = train_test_split(image_data, labels,
                                                    test_size=validation_split, random_state=42, stratify=labels)
train_gen = DataGenerator(time_delay=time_delay).fit(X_train, y_train)
test_gen = DataGenerator(time_delay=time_delay).fit(X_test, y_test)

model = ConvolutionalLstmNN(target_dimensions, channels, target_labels, time_delay=time_delay)
model.fit_generator(train_gen.generate(target_dimensions, batch_size=5),
                    test_gen.generate(target_dimensions, batch_size=5),
                    epochs=10)

## if you want to save a graph of your model layers.
model.save_model_graph()

