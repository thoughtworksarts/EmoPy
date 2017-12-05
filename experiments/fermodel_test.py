import sys
sys.path.append('../fer')
from fermodel import FERModel

target_emotions = ['anger', 'fear', 'calm', 'sad', 'happy', 'surprise']
csv_file_path = "../data/fer2013/fer2013.csv"
model = FERModel(target_emotions, csv_data_path=csv_file_path, raw_dimensions=(48,48), csv_image_col=1, csv_label_col=0, verbose=True)
model.train()
