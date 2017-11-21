import sys
sys.path.append('../feature')
from imageprocessor import ImageProcessor

target_image_dims = (64,64)

d = ImageProcessor()
root_directory = "../data/cohn_kanade_images"
csv_file_path = "../data/fer2013/fer2013.csv"

d.add_feature('hog', {'orientations': 8, 'pixels_per_cell': (16, 16), 'cells_per_block': (1, 1)})
d.add_feature('lbp', {'n_points': 24, 'radius': 3})

fromCsv = True

if fromCsv:
    X_train, y_train, X_test, y_test = d.get_training_data(from_csv=True, dataset_location=csv_file_path, target_image_dims=target_image_dims, initial_image_dims=(48, 48), label_index=0, image_index=1, vector=False, time_series=False)

    print('X_train shape: ' + str(X_train.shape))
    print('y_train shape: ' + str(y_train.shape))
    print('X_test shape: ' + str(X_test.shape))
    print('y_test shape: ' + str(y_test.shape))
else:
    feature_images = d.get_training_data(from_csv=False, dataset_location=root_directory, target_image_dims=target_image_dims, label_index=0, image_index=1, vector=True, time_series=False)

    print(feature_images.shape)
