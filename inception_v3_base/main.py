import sys
sys.path.append('../feature')
sys.path.append('../data')
from dataProcessor import DataProcessor
from transferModel import TransferModel
import cv2
from datetime import datetime
import numpy as np


print('Creating NN with InceptionV3 base model...')
model = TransferModel(model_name='inception_v3')

print('Extracting training data...')

target_image_dims = (128,128)

d = DataProcessor()
root_directory = "../data/cohn_kanade_images"
csv_file_path = "../data/fer2013/fer2013.csv"

d.add_feature('hog', {'orientations': 8, 'pixels_per_cell': (4, 4), 'cells_per_block': (1, 1)})

X_train, y_train, X_test, y_test = d.get_training_data(from_csv=True, dataset_location=csv_file_path, target_image_dims=target_image_dims, initial_image_dims=(48, 48), label_index=0, image_index=1, vector=False, time_series=False)

print('X_train shape: ' + str(X_train.shape))
print('y_train shape: ' + str(y_train.shape))
print('X_test shape: ' + str(X_test.shape))
print('y_test shape: ' + str(y_test.shape))


print ('Training model...')
print('numLayers: ' + str(len(model.model.layers)))
model.fit(X_train, y_train, X_test, y_test)


### ------------ Test ---------------- ###

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame

    start = datetime.now()

    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(gray, target_image_dims, interpolation=cv2.INTER_LINEAR)
    image_3d = np.array([image, image, image]).reshape((target_image_dims[0], target_image_dims[1], 3))

    prediction = model.model.predict(np.array([image_3d]))
    print('Prediction: ' + str(prediction))

    # Display the resulting frame
    cv2.imshow('frame',image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    end = datetime.now()
    print('Singe prediction time: ' + str(end-start))

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
