from fermodel import FERModel

target_emotions = ['anger', 'fear', 'calm', 'sad', 'happy', 'surprise']
csv_file_path = "../data/fer2013/fer2013.csv"
model = FERModel(target_emotions, csv_data_path=csv_file_path, verbose=True)
model.train()
