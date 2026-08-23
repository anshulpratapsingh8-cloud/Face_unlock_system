import cv2

def detect_day_night(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = gray.mean()

    if brightness < 80:
        frame = cv2.convertScaleAbs(frame, alpha=2.5, beta=80)
        mode = "Night"
    else:
        mode = "Day"

    return frame, gray, mode