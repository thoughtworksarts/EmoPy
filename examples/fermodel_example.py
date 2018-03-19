import sys
sys.path.append('../')
from fermodel import FERModel

target_emotions = ['happiness', 'disgust', 'surprise']
model = FERModel(target_emotions, verbose=True)
