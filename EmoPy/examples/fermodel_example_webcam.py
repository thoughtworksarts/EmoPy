# This file allows to perform Emotion detection on frames grabbed from the webcam
# using OpenCV-Python

import cv2
from EmoPy.src.fermodel import FERModel


def capture_image(video_capture, file):

    # Capturing a smaller image fçor speed purposes
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    video_capture.set(cv2.CAP_PROP_FPS, 15)

    if video_capture.isOpened():
        ret = False
        print("Capturing image ...")

        counter = 0
        max_tries = 100
        while not ret and video_capture.isOpened():
            counter += 1
            if counter >= max_tries:
                return None

            # Capture frame-by-frame
            ret, frame = video_capture.read()

        # Save the captured frame on disk
        cv2.imwrite(file, frame)
        print("Image written to: ", file)

    else:
        print("Cannot access the webcam")

    video_capture.release()
    return frame


def display_prediction(frame, frameString, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                       fontScale=1, thickness=2):

    # Display emotion
    retval, _ = cv2.getTextSize(
        frameString, fontFace, fontScale, thickness)
    cv2.rectangle(frame, (0, 0), (20 + retval[0], 50), (0, 0, 0), -1)
    cv2.putText(frame, frameString, (10, 35), fontFace, fontScale,
                (255, 255, 255), thickness, cv2.LINE_AA)

    window_name = 'EmoPy Assessment'
    cv2.imshow(window_name, frame)

    while True:

        key = cv2.waitKey(1)
        # Press Esc to exit the window
        if key == 27 or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    # Closes all windows
    cv2.destroyAllWindows()


def get_emotion_from_camera():

    # Specify the camera which you want to use. The default argument is '0'
    video_capture = cv2.VideoCapture(0)
    file = 'image_data/image.jpg'
    frame = capture_image(video_capture, file)

    if frame is not None:
        # Can choose other target emotions from the emotion subset defined in
                # fermodel.py in src directory. The function
        # defined as `def _check_emotion_set_is_supported(self):`
        target_emotions = ['calm', 'anger', 'happiness']
        model = FERModel(target_emotions, verbose=True)

        frame_string = model.predict(file)
        display_prediction(frame, frame_string)
    else:
        print("Image could not be captured")


if __name__ == '__main__':
    get_emotion_from_camera()
