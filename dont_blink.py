import sys
from PySide6.QtWidgets import QApplication
from blink_monitor import BlinkMonitor

from dotenv import load_dotenv, find_dotenv
import os


load_dotenv(find_dotenv(".env"))
# === RUTAS ===
BASE_DIR = os.environ.get("BASE_DIR", ".")
if BASE_DIR == ".":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

USERS_DIR = os.path.join(BASE_DIR, os.environ.get("USERS_DIR"))
SAMPLES_DIR = os.path.join(BASE_DIR, os.environ.get("SAMPLES_DIR"))

os.makedirs(USERS_DIR, exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BlinkMonitor(USERS_DIR, SAMPLES_DIR)
    window.resize(1280, 720)
    window.show()
    sys.exit(app.exec())