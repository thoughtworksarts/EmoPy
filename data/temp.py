import sys
sys.path.append('../feature')
from dataProcessor import DataProcessor


d = DataProcessor()
root_directory = "../data/cohn_kanade_images"
csv_file_path = "../data/fer2013/fer2013.csv"
# feature_images = d.get_time_series_image_feature_array_from_directory(root_directory)
feature_images = d.get_image_feature_array_from_csv(csv_file_path)

print(feature_images.shape)