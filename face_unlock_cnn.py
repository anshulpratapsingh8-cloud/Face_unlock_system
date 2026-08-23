import cv2
import os
import numpy as np
from tkinter import *
from tkinter import messagebox
from PIL import Image, ImageTk
from deepface import DeepFace
import mediapipe as mp

from light_sensor import detect_day_night
from face_detector import detect_faces

# 🔐 PIN
USER_PIN = "1234"

# 👁️ MediaPipe setup
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh()

# 👁️ Eye landmark indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

blink_counter = 0
blink_detected = False

cam = cv2.VideoCapture(0)

unlock_counter = 0
frame_count = 0
pin_open = False

window = Tk()
window.title("CNN Face Unlock System")
window.geometry("450x550")
window.configure(bg="black")

Label(window, text="Align your face",
      fg="white", bg="black", font=("Arial", 14)).pack(pady=10)

camera_label = Label(window)
camera_label.pack()

status_label = Label(window, text="Camera Ready",
                     fg="gray", bg="black", font=("Arial", 12))
status_label.pack(pady=5)


# 👁️ Eye ratio function
def eye_ratio(landmarks, eye):
    p1 = np.array([landmarks[eye[0]].x, landmarks[eye[0]].y])
    p2 = np.array([landmarks[eye[1]].x, landmarks[eye[1]].y])
    p3 = np.array([landmarks[eye[2]].x, landmarks[eye[2]].y])
    p4 = np.array([landmarks[eye[3]].x, landmarks[eye[3]].y])
    p5 = np.array([landmarks[eye[4]].x, landmarks[eye[4]].y])
    p6 = np.array([landmarks[eye[5]].x, landmarks[eye[5]].y])

    vertical = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)

    return vertical / (2.0 * horizontal)


# 👁️ REAL blink detection
def detect_blink(frame):
    global blink_counter, blink_detected

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if result.multi_face_landmarks:
        landmarks = result.multi_face_landmarks[0].landmark

        left = eye_ratio(landmarks, LEFT_EYE)
        right = eye_ratio(landmarks, RIGHT_EYE)

        ear = (left + right) / 2

        if ear < 0.20:   # eye closed
            blink_counter += 1
        else:
            if blink_counter > 2:
                blink_detected = True
            blink_counter = 0

    return blink_detected


# 🔐 PIN window
def ask_pin():
    global pin_open

    if pin_open:
        return

    pin_open = True

    pin_window = Toplevel()
    pin_window.title("Enter PIN")
    pin_window.geometry("250x150")
    pin_window.grab_set()

    Label(pin_window, text="Enter PIN").pack(pady=5)

    entry = Entry(pin_window, show="*")
    entry.pack(pady=5)

    def check():
        global pin_open

        if entry.get() == USER_PIN:
            messagebox.showinfo("Success", "Unlocked via PIN 🔓")
            cam.release()
            window.destroy()
        else:
            messagebox.showerror("Error", "Wrong PIN")
            pin_open = False

    def on_close():
        global pin_open
        pin_open = False
        pin_window.destroy()

    Button(pin_window, text="Submit", command=check).pack(pady=10)
    pin_window.protocol("WM_DELETE_WINDOW", on_close)


def update_camera():
    global unlock_counter, frame_count, blink_detected

    ret, frame = cam.read()

    if ret:

        frame, gray, mode = detect_day_night(frame)

        live = "Live" if detect_blink(frame) else "Fake"

        faces = detect_faces(gray)

        for (x, y, w, h) in faces:

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

            face = frame[y:y+h, x:x+w]
            face = cv2.resize(face, (160,160))   # ⚡ speed fix

            frame_count += 1

            # ⚡ less frequent check (lag fix)
            if frame_count % 25 == 0:

                try:
                    temp_path = "temp.jpg"
                    cv2.imwrite(temp_path, face)

                    result = DeepFace.find(
                        img_path=temp_path,
                        db_path="dataset",
                        enforce_detection=False,
                        model_name="Facenet512"
                    )

                    if os.path.exists(temp_path):
                        os.remove(temp_path)

                    if len(result) > 0 and len(result[0]) > 0:

                        distance = result[0].iloc[0]['distance']
                        print("Distance:", distance)

                        # 🔒 strict match + blink required
                        if distance < 0.30 and live == "Live":

                            unlock_counter += 1
                            status_label.config(text=f"Verifying {unlock_counter}", fg="yellow")

                            if unlock_counter > 2:
                                status_label.config(text="Unlocked", fg="green")

                                messagebox.showinfo("Success", "Face Recognized 🔓")

                                blink_detected = False  # reset

                                cam.release()
                                window.destroy()
                                return

                        else:
                            unlock_counter = 0
                            status_label.config(text="Face Failed - Enter PIN", fg="red")
                            ask_pin()

                    else:
                        unlock_counter = 0
                        status_label.config(text="No Match - Enter PIN", fg="red")
                        ask_pin()

                except Exception as e:
                    print("Error:", e)
                    status_label.config(text="Scanning...", fg="orange")

        cv2.putText(frame, mode, (20,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)

        camera_label.imgtk = imgtk
        camera_label.configure(image=imgtk)

    window.after(10, update_camera)


update_camera()
window.mainloop()

cam.release()
cv2.destroyAllWindows()