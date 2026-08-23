import cv2
import numpy as np

def check_liveness(prev_gray, curr_gray):

    if prev_gray is None:
        return "Checking"

    diff = cv2.absdiff(prev_gray, curr_gray)
    non_zero = np.sum(diff > 25)

    # 🔥 FIX: threshold kam kiya
    if non_zero > 2000:
        return "Live"
    else:
        return "Fake"