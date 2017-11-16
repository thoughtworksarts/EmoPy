import cv2
from datetime import datetime
import numpy as np
from keras.models import load_model

cap = cv2.VideoCapture(0)
model = load_model('../trained_models/inception_v3_model_1.h5')
target_image_dims = (128,128)

while(True):

    start = datetime.now()

    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(gray, target_image_dims, interpolation=cv2.INTER_LINEAR)
    image_3d = np.array([image, image, image]).reshape((target_image_dims[0], target_image_dims[1], 3))

    print(image_3d[0][0])

    prediction = model.predict(np.array([image_3d]))
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