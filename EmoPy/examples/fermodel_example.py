import sys
sys.path.append('../')
from src.fermodel import FERModel

target_emotions = ['calm', 'anger', 'happiness']
model = FERModel(target_emotions, verbose=True)

print('Predicting on happy image...')
model.predict('image_data/sample_happy_image.png')

print('Predicting on disgust image...')
model.predict('image_data/sample_disgust_image.png')

print('Predicting on anger image...')
model.predict('image_data/sample_anger_image2.png')
