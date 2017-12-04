from fermodel import FERModel

target_emotions = ['anger', 'fear', 'calm', 'sad', 'happy', 'surprise']
csv_file_path = "../data/fer2013/fer2013.csv"
model = FERModel(target_emotions, data_path=csv_file_path, extract_features=True, verbose=True)
model.train()