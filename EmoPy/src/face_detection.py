import cv2

class FaceDetector:
    def __init__(self, scaleFactor=1.1, minNeighbors=5, minSize=(30,30)):
        self.scaleFactor = scaleFactor
        self.minNeighbors = minNeighbors
        self.minSize = minSize

    def _detect_faces(self, image):
        faceCascade = cv2.CascadeClassifier('../docs/haarcascade_frontalface_default.xml')
        return faceCascade.detectMultiScale(
            image,
            scaleFactor=self.scaleFacotr,
            minNeighbors=self.minNeighbors,
            minSize=self.minSize
        )

    def crop_face(image, defaultFirst=True):
        faces = self._detect_faces(image)
        if (len(faces) > 1 and defaultFirst) or len(faces) is 1:
            (x, y, w, h) = faces[0]
            return image[y:y+h, x:x+w]
        print("{numFaces} faces were found in image. Not cropping".format(numFaces=len(faces)))
        return image
