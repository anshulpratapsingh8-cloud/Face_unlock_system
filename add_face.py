import cv2
import os

cam = cv2.VideoCapture(0)

face_detector = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml'
)

path = 'dataset/user1'
os.makedirs(path, exist_ok=True)

count = 0

print("Look at camera... Move face slowly")

while True:
    ret, frame = cam.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:

        # blur check (important 🔥)
        face_img = gray[y:y+h, x:x+w]
        blur = cv2.Laplacian(face_img, cv2.CV_64F).var()

        if blur > 50:   # only clear images save
            count += 1

            cv2.imwrite(f"{path}/{count}.jpg", face_img)

            cv2.putText(frame, f"Saved: {count}", (x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0,255,0), 2)

        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

    cv2.imshow("Capture Face", frame)

    # slow capture (important)
    cv2.waitKey(100)

    if cv2.waitKey(1) == 27 or count >= 60:
        break

cam.release()
cv2.destroyAllWindows()

print("Face data collected successfully")