from EmoPy.src.fermodel import FERModel
from pkg_resources import resource_filename

target_emotions = ['calm', 'anger', 'happiness']
model = FERModel(target_emotions, verbose=True)

print('Predicting on happy image...')
model.predict(resource_filename('EmoPy.examples','image_data/sample_happy_image.png'))

print('Predicting on disgust image...')
model.predict(resource_filename('EmoPy.examples','image_data/sample_disgust_image.png'))

print('Predicting on anger image...')
model.predict(resource_filename('EmoPy.examples','image_data/sample_anger_image2.png'))
