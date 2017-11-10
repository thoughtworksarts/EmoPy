import sys
sys.path.append('../feature')
from dataProcessor import DataProcessor


d = DataProcessor()
root_directory = "../data/cohn_kanade_images"
csv_file_path = "../data/fer2013/fer2013.csv"

d.add_feature('hog', {'orientations': 8, 'pixels_per_cell': (16, 16), 'cells_per_block': (1, 1)})
d.add_feature('lbp', {'n_points': 24, 'radius': 3})

# feature_images = d.get_image_features(from_csv=True, dataset_location=csv_file_path, target_image_dims=(128,128), initial_image_dims=(48, 48), label_index=0, image_index=1, vector=True, time_series=False)

feature_images = d.get_image_features(from_csv=False, dataset_location=root_directory, target_image_dims=(128,128), label_index=0, image_index=1, vector=True, time_series=False)

print(feature_images.shape)
# print(feature_images)
