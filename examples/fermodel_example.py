import sys
sys.path.append('../')
from fermodel import FERModel

target_emotions = ['anger', 'fear', 'surprise', 'calm']
model = FERModel(target_emotions, verbose=True)

model.predict('image_data/sample_image.jpg')
