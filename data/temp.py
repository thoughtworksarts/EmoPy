import sys
sys.path.append('../feature')
from dataProcessor import DataProcessor


d = DataProcessor()
root_directory = "../data/cohn_kanade_images"
feature_images = d.get_time_series_image_feature_array(root_directory)

print(feature_images.shape)