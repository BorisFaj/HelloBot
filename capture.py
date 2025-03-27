import cv2
import numpy as np
import time
from mss import mss
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(".env"))

class Capturer:
    def __init__(self, source: str, samples_dir: str):
        self.samples_dir = samples_dir
        self.sct = mss()
        self.monitor = self.sct.monitors[1]
        self.cap = None
        self.reload_default_cap(source)

    def save_face_crop(self, face_crop, encoding):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        _filename = os.path.join(self.samples_dir, f"face_{timestamp}.jpg")
        encoding_path = os.path.join(self.samples_dir, f"face_{timestamp}.npy")
        print(f"Saving crop to {_filename}")
        cv2.imwrite(_filename, cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR))
        np.save(encoding_path, encoding)

    def reload_default_cap(self, source):
        self.cap = cv2.VideoCapture(0) if source == "cam" else None

    def get_cap(self):
        return self.cap

    def release_cap(self):
        self.cap.release()

    def get_frame(self, source):
        if source == "screen":
            img = np.array(self.sct.grab(self.monitor))
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        else:
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(0)
            ret, frame = self.cap.read()
            return frame if ret else None

    def eye_aspect_ratio(self, coords):
        A = np.linalg.norm(np.array(coords[1]) - np.array(coords[5]))
        B = np.linalg.norm(np.array(coords[2]) - np.array(coords[4]))
        C = np.linalg.norm(np.array(coords[0]) - np.array(coords[3]))
        return (A + B) / (2.0 * C)
