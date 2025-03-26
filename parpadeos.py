import sys
import os
import cv2
import numpy as np
import mediapipe as mp
import time
from mss import mss
from collections import defaultdict, deque
from PySide6.QtWidgets import (QApplication, QLabel, QMainWindow, QVBoxLayout,
                               QWidget, QHBoxLayout, QListWidget, QListWidgetItem,
                               QPushButton)
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap, QIcon
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import face_recognition

# === CONFIGURACIÓN ===
SOURCE = "screen"  # "cam" o "screen"
EAR_THRESH = 0.26
NUM_FRAMES = 2
WINDOW_SECONDS = 60
ENCODING_TOLERANCE = 0.6
ENABLE_CROPS = True

# === RUTAS ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USERS_DIR = os.path.join(BASE_DIR, "data/users")
SAMPLES_DIR = os.path.join(BASE_DIR, "data/sample_crops")

os.makedirs(USERS_DIR, exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)

sct = mss()
monitor = sct.monitors[1]
cap = cv2.VideoCapture(0) if SOURCE == "cam" else None

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=5)
index_left_eye = [33, 160, 158, 133, 153, 144]
index_right_eye = [362, 385, 387, 263, 373, 380]

blink_times = defaultdict(deque)
aux_counters = defaultdict(int)
emotion_state = defaultdict(str)
user_embeddings = {}  # nombre_usuario -> [encoding1, encoding2, ...]
face_colors = {}

def load_user_embeddings():
    global user_embeddings
    user_embeddings.clear()
    for user_dir in os.listdir(USERS_DIR):
        user_path = os.path.join(USERS_DIR, user_dir)
        encodings = []
        for file in os.listdir(user_path):
            if file.endswith(".npy"):
                try:
                    data = np.load(os.path.join(user_path, file))
                    if data.ndim == 1:
                        encodings.append(data)
                    elif data.ndim == 2:
                        encodings.extend(data)
                except Exception as e:
                    print(f"Error loading {file}: {e}")
        if encodings:
            user_embeddings[user_dir] = encodings
            if user_dir not in face_colors:
                face_colors[user_dir] = tuple(np.random.randint(0, 255, size=3).tolist())

def save_face_crop(face_crop, encoding):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(SAMPLES_DIR, f"face_{timestamp}.jpg")
    encoding_path = os.path.join(SAMPLES_DIR, f"face_{timestamp}.npy")
    print(f"Saving crop to {filename}")
    cv2.imwrite(filename, cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR))
    np.save(encoding_path, encoding)

def setup_buttons(parent, layout):
    parent.toggle_source_btn = QPushButton()
    parent.toggle_source_btn.setIcon(QIcon.fromTheme("camera-video"))
    parent.toggle_source_btn.setToolTip("Cambiar fuente (cam/screen)")
    parent.toggle_source_btn.setFixedSize(30, 30)
    parent.toggle_source_btn.clicked.connect(parent.toggle_source)

    parent.toggle_crop_btn = QPushButton()
    parent.toggle_crop_btn.setIcon(QIcon.fromTheme("media-record"))
    parent.toggle_crop_btn.setCheckable(True)
    parent.toggle_crop_btn.setChecked(ENABLE_CROPS)
    parent.toggle_crop_btn.setFixedSize(30, 30)
    parent.toggle_crop_btn.setToolTip("Guardar muestras")

    parent.save_all_btn = QPushButton("Guardar todas las caras")
    parent.save_all_btn.setFixedSize(150, 30)
    parent.save_all_btn.clicked.connect(parent.save_all_faces)

    def update_crop_icon():
        global ENABLE_CROPS
        ENABLE_CROPS = parent.toggle_crop_btn.isChecked()
        if ENABLE_CROPS:
            parent.toggle_crop_btn.setStyleSheet("background-color: #66ff66;")
            print("Crops activados")
        else:
            parent.toggle_crop_btn.setStyleSheet("background-color: #ff6666;")
            print("Crops desactivados")

    parent.toggle_crop_btn.clicked.connect(update_crop_icon)
    update_crop_icon()

    layout.addWidget(parent.toggle_source_btn)
    layout.addWidget(parent.toggle_crop_btn)
    layout.addWidget(parent.save_all_btn)

def get_frame():
    global cap
    if SOURCE == "screen":
        img = np.array(sct.grab(monitor))
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    else:
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)
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
        load_user_embeddings()

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

        button_layout = QHBoxLayout()
        setup_buttons(self, button_layout)
        video_layout.addLayout(button_layout)

        main_layout = QHBoxLayout()
        main_layout.addLayout(video_layout, 2)
        main_layout.addWidget(self.timeline, 1)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(40)

        self.last_user_dirs = set(os.listdir(USERS_DIR))
        self.detected_faces = []

    def toggle_source(self):
        global SOURCE, cap
        SOURCE = "cam" if SOURCE == "screen" else "screen"
        if cap:
            cap.release()
        cap = cv2.VideoCapture(0) if SOURCE == "cam" else None

    def toggle_crops(self):
        global ENABLE_CROPS
        ENABLE_CROPS = not ENABLE_CROPS
        self.log_event("GUI", f"Crops = {ENABLE_CROPS}")

    def save_all_faces(self):
        for (face_crop, encoding) in self.detected_faces:
            save_face_crop(face_crop, encoding)
        print(f"Guardadas {len(self.detected_faces)} caras actuales.")

    def update_frame(self):
        frame = get_frame()
        if frame is None:
            return

        current_dirs = set(os.listdir(USERS_DIR))
        if current_dirs != self.last_user_dirs:
            load_user_embeddings()
            self.last_user_dirs = current_dirs

        height, width, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        face_locations_full = face_recognition.face_locations(rgb)

        current_time = time.time()
        self.timestamps.append(current_time)
        self.detected_faces.clear()

        self.ax.clear()
        for landmarks in results.multi_face_landmarks or []:
            xs = [lm.x * width for lm in landmarks.landmark]
            ys = [lm.y * height for lm in landmarks.landmark]
            x0, y0 = int(min(xs)), int(min(ys))
            x1, y1 = int(max(xs)), int(max(ys))

            if (y1 - y0) < 40 or (x1 - x0) < 40:
                continue

            matched_face = None
            for loc in face_locations_full:
                top, right, bottom, left = loc
                if left <= x0 <= right and top <= y0 <= bottom:
                    matched_face = loc
                    break

            face_id = "Unknown"
            if matched_face:
                encoding = face_recognition.face_encodings(rgb, known_face_locations=[matched_face])[0]
                for user, enc_list in user_embeddings.items():
                    distances = face_recognition.face_distance(enc_list, encoding)
                    if len(distances) > 0 and np.min(distances) < ENCODING_TOLERANCE:
                        face_id = user
                        break
                top, right, bottom, left = matched_face
                face_crop = rgb[top:bottom, left:right]
                if face_crop.size > 0:
                    self.detected_faces.append((face_crop, encoding))
                    if ENABLE_CROPS:
                        save_face_crop(face_crop, encoding)

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

                emotion = "unknown"
                if emotion != emotion_state[face_id]:
                    self.log_event(face_id, f"Emotion: {emotion}")
                    emotion_state[face_id] = emotion

                color = face_colors.get(face_id, (0, 255, 0))
                cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
                draw_bar(frame, len(blink_times[face_id]), bpm, x0, y1 + 15, emotion, face_id)
                self.data[face_id].append((current_time, len(blink_times[face_id]), bpm))

        for face_id, entries in self.data.items():
            if len(entries) < 2:
                continue
            times, blink_counts, bpm_values = zip(*entries)
            self.ax.plot(times, bpm_values, label=f"{face_id}")

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