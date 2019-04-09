# This file allows to perform Emotion detection on frames grabbed from the webcam using OpenCV-Python

import cv2
from EmoPy.src.fermodel import FERModel

fontFace = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
thickness = 2

# Can choose other target emotions from the emotion subset defined in fermodel.py in src directory. The function
# defined as `def _check_emotion_set_is_supported(self):`
target_emotions = ['anger', 'happiness']
target_emotion_indices = '03'
model = FERModel(target_emotions, 'models/conv_model_%s.hdf5' % target_emotion_indices, verbose=True)

# Specify the camera which you want to use. The default argument is '0'
video_capture = cv2.VideoCapture(0)
# Capturing a smaller image fçor speed purposes
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
video_capture.set(cv2.CAP_PROP_FPS, 15)

while True:
	ret = False
	while (ret is False):
		ret, frame = video_capture.read()

	frameString = model.predict_from_ndarray(frame)

	retval, baseline = cv2.getTextSize(frameString, fontFace, fontScale, thickness)
	cv2.rectangle(frame, (0, 0), (20 + retval[0], 50), (0, 0, 0), -1)
	cv2.putText(frame, frameString, (10, 35), fontFace, fontScale, (255, 255, 255), thickness, cv2.LINE_AA)
	cv2.imshow('Video', frame)
	cv2.waitKey(1)

	if cv2.waitKey(1) & 0xFF == 27:
		break

cv2.destroyAllWindows()
