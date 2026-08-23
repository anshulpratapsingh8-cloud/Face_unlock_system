import cv2

face_cascade = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml'
)

def detect_faces(gray):
    return face_cascade.detectMultiScale(gray, 1.3, 5)