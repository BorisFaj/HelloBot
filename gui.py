import cv2
from PySide6.QtWidgets import QPushButton
from PySide6.QtGui import QIcon


def setup_buttons(parent, layout, enable_crops):
    parent.toggle_source_btn = QPushButton()
    parent.toggle_source_btn.setIcon(QIcon.fromTheme("camera-video"))
    parent.toggle_source_btn.setToolTip("Cambiar fuente (cam/screen)")
    parent.toggle_source_btn.setFixedSize(30, 30)
    parent.toggle_source_btn.clicked.connect(parent.toggle_source)

    parent.toggle_crop_btn = QPushButton()
    parent.toggle_crop_btn.setIcon(QIcon.fromTheme("media-record"))
    parent.toggle_crop_btn.setCheckable(True)
    parent.toggle_crop_btn.setChecked(enable_crops)
    parent.toggle_crop_btn.setFixedSize(30, 30)
    parent.toggle_crop_btn.setToolTip("Guardar muestras")

    parent.save_all_btn = QPushButton("Guardar todas las caras")
    parent.save_all_btn.setFixedSize(150, 30)
    parent.save_all_btn.clicked.connect(parent.save_all_faces)

    def update_crop_icon():
        global enable_crops
        enable_crops = parent.toggle_crop_btn.isChecked()
        if enable_crops:
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



def draw_bar(frame, blink_count, bpm, x, y, emotion, face_id, face_colors):
    bar_width = 100
    filled = int(min(bpm / 60 * bar_width, bar_width))
    color = face_colors.get(face_id, (0, 255, 0))
    cv2.putText(frame, f"ID: {face_id}", (x, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.putText(frame, f"Blinks: {blink_count}", (x, y - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    cv2.putText(frame, f"{bpm:.1f} bpm", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.putText(frame, f"{emotion}", (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
    cv2.rectangle(frame, (x, y), (x + bar_width, y + 10), (100,100,100), 1)
    cv2.rectangle(frame, (x, y), (x + filled, y + 10), color, -1)