#This file allows to perform Emotion detection on frames grabbed from the webcam using OpenCV-Python

import cv2
import sys
sys.path.append('../')

#Choose the type of Face Expression Model
from src.fermodel import FERModel

#Frame Number
FRAME_NUM = 0

#Choose the type of face detector cascade you want to use
cascPath = "~/EmoPy/venv/lib/python3.5/site-packages/cv2/data/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
#Specify the camera which you want to use. The default argument is '0'
video_capture = cv2.VideoCapture(0)

while True:
	#Capture frame-by-frame
	ret, frame = video_capture.read()
	#Save the captured frame on disk
	file = '~/EmoPy/models/examples/image_data/image.jpg'
	cv2.imwrite(file, frame)
	#Can choose other target emotions from the emotion subset defined in fermodel.py in src directory. The function
	# defined as `def _check_emotion_set_is_supported(self):`
	target_emotions = ['calm', 'anger', 'happiness']
	model = FERModel(target_emotions, verbose=True)
	frameString = model.predict(file)
	#Display frame number and emotion 
	cv2.putText(frame, 'Frame:' + str(FRAME_NUM), (10, 40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 3, (0, 0, 255), 2, cv2.LINE_AA)
	cv2.putText(frame, frameString, (10,450), cv2.FONT_HERSHEY_COMPLEX_SMALL, 3, (0,255,0), 2, cv2.LINE_AA)
	cv2.imshow('Video', frame)
	cv2.waitKey(1)
	FRAME_NUM += 1
	#Press Esc to exit the window
	if cv2.waitKey(1) & 0xFF == 27:
		break
#Closes all windows
cv2.destroyAllWindows()

