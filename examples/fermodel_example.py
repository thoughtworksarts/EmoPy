import sys
sys.path.append('../')
from fermodel import FERModel

emotion_map = fer_dataset_label_map = {'0': 'anger', '2': 'fear'}
csv_file_path = "image_data/sample.csv"
model = FERModel(fer_dataset_label_map, csv_data_path=csv_file_path, raw_dimensions=(48,48), csv_image_col=1, csv_label_col=0, verbose=True)
model.train()
