import sys
import cv2
import numpy as np
import mediapipe as mp
import time
from mss import mss
from collections import defaultdict, deque
from PySide6.QtWidgets import (QApplication, QLabel, QMainWindow, QVBoxLayout,
                               QWidget, QHBoxLayout, QListWidget, QListWidgetItem)
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from deepface import DeepFace
import face_recognition
import random

# === CONFIGURACIÓN ===
SOURCE = "cam"  # "cam" o "screen"
EAR_THRESH = 0.26
NUM_FRAMES = 2
WINDOW_SECONDS = 60
ENCODING_TOLERANCE = 0.45

sct = mss()
monitor = sct.monitors[1]
cap = cv2.VideoCapture(0) if SOURCE == "cam" else None

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=10)
index_left_eye = [33, 160, 158, 133, 153, 144]
index_right_eye = [362, 385, 387, 263, 373, 380]

blink_times = defaultdict(deque)
aux_counters = defaultdict(int)
emotion_state = defaultdict(str)
known_encodings = []
known_ids = []
next_face_id = 0
face_colors = {}

# === FUNCIONES ===
def get_frame():
    if SOURCE == "screen":
        img = np.array(sct.grab(monitor))
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    else:
        ret, frame = cap.read()
        return frame if ret else None

def eye_aspect_ratio(coords):
    A = np.linalg.norm(np.array(coords[1]) - np.array(coords[5]))
    B = np.linalg.norm(np.array(coords[2]) - np.array(coords[4]))
    C = np.linalg.norm(np.array(coords[0]) - np.array(coords[3]))
    return (A + B) / (2.0 * C)

def draw_bar(frame, blink_count, bpm, x, y, emotion, face_id):
    bar_width = 100
    filled = int(min(bpm / 60 * bar_width, bar_width))
    color = face_colors.get(face_id, (0, 255, 0))
    cv2.putText(frame, f"ID: {face_id}", (x, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.putText(frame, f"Blinks: {blink_count}", (x, y - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    cv2.putText(frame, f"{bpm:.1f} bpm", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.putText(frame, f"{emotion}", (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
    cv2.rectangle(frame, (x, y), (x + bar_width, y + 10), (100,100,100), 1)
    cv2.rectangle(frame, (x, y), (x + filled, y + 10), color, -1)

class BlinkMonitor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Blink Frequency Monitor")

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)

        self.timeline = QListWidget()

        self.figure = Figure(figsize=(5, 3))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.data = defaultdict(list)
        self.timestamps = []

        video_layout = QVBoxLayout()
        video_layout.addWidget(self.video_label)
        video_layout.addWidget(self.canvas)

        main_layout = QHBoxLayout()
        main_layout.addLayout(video_layout, 2)
        main_layout.addWidget(self.timeline, 1)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def update_frame(self):
        global next_face_id
        frame = get_frame()
        if frame is None:
            return

        height, width, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        current_time = time.time()
        self.timestamps.append(current_time)

        self.ax.clear()
        for landmarks in results.multi_face_landmarks or []:
            xs = [lm.x * width for lm in landmarks.landmark]
            ys = [lm.y * height for lm in landmarks.landmark]
            x0, y0 = int(min(xs)), int(min(ys))
            x1, y1 = int(max(xs)), int(max(ys))

            if (y1 - y0) < 20 or (x1 - x0) < 20:
                continue

            face_crop = rgb[y0:y1, x0:x1]
            if face_crop.size == 0:
                continue

            face_id = None
            try:
                encoding = face_recognition.face_encodings(face_crop)
                if encoding:
                    encoding = encoding[0]
                    distances = face_recognition.face_distance(known_encodings, encoding)
                    if len(distances) > 0 and np.min(distances) < ENCODING_TOLERANCE:
                        face_id = known_ids[np.argmin(distances)]
                    else:
                        face_id = next_face_id
                        known_encodings.append(encoding)
                        known_ids.append(face_id)
                        face_colors[face_id] = tuple(np.random.randint(0, 255, size=3).tolist())
                        next_face_id += 1
            except Exception as e:
                print("Encoding error:", e)
                face_id = f"temp_{x0}_{y0}"
                if face_id not in face_colors:
                    face_colors[face_id] = tuple(np.random.randint(0, 255, size=3).tolist())

            left_eye = []
            right_eye = []
            for idx in index_left_eye:
                x = int(landmarks.landmark[idx].x * width)
                y = int(landmarks.landmark[idx].y * height)
                left_eye.append([x, y])
            for idx in index_right_eye:
                x = int(landmarks.landmark[idx].x * width)
                y = int(landmarks.landmark[idx].y * height)
                right_eye.append([x, y])

            if len(left_eye) == 6 and len(right_eye) == 6:
                ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2
                if ear < EAR_THRESH:
                    aux_counters[face_id] += 1
                else:
                    if aux_counters[face_id] >= NUM_FRAMES:
                        blink_times[face_id].append(current_time)
                        self.log_event(face_id, "Blink")
                    aux_counters[face_id] = 0

                blink_times[face_id] = deque([t for t in blink_times[face_id] if current_time - t <= WINDOW_SECONDS])
                bpm = len(blink_times[face_id]) * (60 / WINDOW_SECONDS)

                try:
                    analysis = DeepFace.analyze(face_crop, actions=["emotion"], enforce_detection=False)
                    emotion = analysis[0]["dominant_emotion"]
                except Exception as e:
                    print("DeepFace error:", e)
                    emotion = "unknown"

                if emotion != emotion_state[face_id]:
                    self.log_event(face_id, f"Emotion: {emotion}")
                    emotion_state[face_id] = emotion

                color = face_colors.get(face_id, (0, 255, 0))
                cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
                draw_bar(frame, len(blink_times[face_id]), bpm, x0, y1 + 15, emotion, face_id)
                self.data[face_id].append((current_time, len(blink_times[face_id]), bpm))

        for face_id, entries in self.data.items():
            times, blink_counts, bpm_values = zip(*entries)
            self.ax.plot(times, bpm_values, label=f"Face {face_id}")

        self.ax.set_title("Blinks per Minute")
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("BPM")
        self.ax.legend()
        self.canvas.draw()

        display = cv2.resize(frame, (640, 360))
        image = QImage(display.data, display.shape[1], display.shape[0], QImage.Format_BGR888)
        self.video_label.setPixmap(QPixmap.fromImage(image))

    def log_event(self, face_id, event_type):
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        text = f"[{timestamp}] Face {face_id}: {event_type}"
        self.timeline.addItem(QListWidgetItem(text))
        self.timeline.scrollToBottom()

    def closeEvent(self, event):
        if cap:
            cap.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BlinkMonitor()
    window.resize(1280, 720)
    window.show()
    sys.exit(app.exec())
