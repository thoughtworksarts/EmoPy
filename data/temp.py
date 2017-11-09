import sys
sys.path.append('../feature')
from dataProcessor import DataProcessor


d = DataProcessor()
root_directory = "../data/cohn_kanade_images"
csv_file_path = "../data/fer2013/fer2013.csv"
# feature_images = d.get_time_series_image_feature_array_from_directory(root_directory, None)
feature_images = d.get_image_features(from_csv=True, dataset_location=csv_file_path, initial_image_dims=(48, 48), target_image_dims=(64,64), feature_set=['hog'], label_index=0, image_index=1, vector=True, time_series=False)
# feature_images = d.get_image_features(from_csv=True, dataset_location=csv_file_path, initial_image_dims=(48, 48), target_image_dims=(64,64), feature_set=['hog'], label_index=0, image_index=1, vector=True, time_series=False)

print(feature_images.shape)
# print(feature_images)
