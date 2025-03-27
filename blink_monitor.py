import os
import cv2
import numpy as np
import time
from PySide6.QtWidgets import (QLabel, QMainWindow, QVBoxLayout,
                               QWidget, QHBoxLayout, QListWidget, QListWidgetItem)
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import face_recognition
from user import embeddings
import gui
from capture import Capturer
import mediapipe as mp
from collections import defaultdict, deque

class BlinkMonitor(QMainWindow):
    def __init__(self, users_dir: str, samples_dir: str):
        super().__init__()
        self.users_dir = users_dir
        self.samples_dir = samples_dir
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=5)
        self.index_left_eye = [33, 160, 158, 133, 153, 144]
        self.index_right_eye = [362, 385, 387, 263, 373, 380]

        self.blink_times = defaultdict(deque)
        self.aux_counters = defaultdict(int)
        self.emotion_state = defaultdict(str)
        self.user_embeddings, self.face_colors = embeddings.load_user_embeddings(self.users_dir)

        self.SOURCE = os.environ.get("SOURCE")
        self.EAR_THRESH = float(os.environ.get("EAR_THRESH"))
        self.NUM_FRAMES = int(os.environ.get("NUM_FRAMES"))
        self.WINDOW_SECONDS = int(os.environ.get("WINDOW_SECONDS"))
        self.ENCODING_TOLERANCE = float(os.environ.get("ENCODING_TOLERANCE"))
        self.ENABLE_CROPS = bool(os.environ.get("ENABLE_CROPS"))

        self.cap = Capturer(self.SOURCE, self.samples_dir)

        # GUI
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

        button_layout = QHBoxLayout()
        gui.setup_buttons(self, button_layout, self.ENABLE_CROPS)
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

        self.last_user_dirs = set(os.listdir(self.users_dir))
        self.detected_faces = []

    def toggle_source(self):
        self.SOURCE = "cam" if self.SOURCE == "screen" else "screen"
        if self.cap.get_cap():
            self.cap.release_cap()
        self.cap.reload_default_cap(self.SOURCE)

    def toggle_crops(self):
        self.ENABLE_CROPS = not self.ENABLE_CROPS
        self.log_event("GUI", f"Crops = {self.ENABLE_CROPS}")

    def save_all_faces(self):
        for (face_crop, encoding) in self.detected_faces:
            self.cap.save_face_crop(face_crop, encoding)
        print(f"Guardadas {len(self.detected_faces)} caras actuales.")

    def update_embeddings(self):
        current_dirs = set(os.listdir(self.users_dir))
        if current_dirs != self.last_user_dirs:
            self.user_embeddings, self.face_colors = embeddings.load_user_embeddings(self.users_dir)
            self.last_user_dirs = current_dirs

    def update_frame(self):
        frame = self.cap.get_frame(self.SOURCE)
        if frame is None:
            return

        self.update_embeddings()

        height, width, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
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
                for user, enc_list in self.user_embeddings.items():
                    distances = face_recognition.face_distance(enc_list, encoding)
                    if len(distances) > 0 and np.min(distances) < self.ENCODING_TOLERANCE:
                        face_id = user
                        break
                top, right, bottom, left = matched_face
                face_crop = rgb[top:bottom, left:right]
                if face_crop.size > 0:
                    self.detected_faces.append((face_crop, encoding))
                    if self.ENABLE_CROPS:
                        self.cap.save_face_crop(face_crop, encoding)

            left_eye = []
            right_eye = []
            for idx in self.index_left_eye:
                x = int(landmarks.landmark[idx].x * width)
                y = int(landmarks.landmark[idx].y * height)
                left_eye.append([x, y])
            for idx in self.index_right_eye:
                x = int(landmarks.landmark[idx].x * width)
                y = int(landmarks.landmark[idx].y * height)
                right_eye.append([x, y])

            if len(left_eye) == 6 and len(right_eye) == 6:
                ear = (self.cap.eye_aspect_ratio(left_eye) + self.cap.eye_aspect_ratio(right_eye)) / 2
                if ear < self.EAR_THRESH:
                    self.aux_counters[face_id] += 1
                else:
                    if self.aux_counters[face_id] >= self.NUM_FRAMES:
                        self.blink_times[face_id].append(current_time)
                        self.log_event(face_id, "Blink")
                    self.aux_counters[face_id] = 0

                self.blink_times[face_id] = deque([t for t in self.blink_times[face_id] if current_time - t <= self.WINDOW_SECONDS])
                bpm = len(self.blink_times[face_id]) * (60 / self.WINDOW_SECONDS)

                color = self.face_colors.get(face_id, (0, 255, 0))
                cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
                gui.draw_bar(frame, len(self.blink_times[face_id]), bpm, x0, y1 + 15, "emotion", face_id, self.face_colors)
                self.data[face_id].append((current_time, len(self.blink_times[face_id]), bpm))

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
        if self.cap.get_cap():
            self.cap.release_cap()
        event.accept()
