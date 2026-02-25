"""
Configuration — Coconut Detection System
Uses YOLOv8 for real-time coconut detection via WiFi cameras.
"""

# ─── Cameras ─────────────────────────────────────────────────────────────────
CAMERA_SOURCES = [
    {"id": "cam1", "label": "Camera 1", "url": "rtsp://Banana12a:98819881xyz@192.168.43.9:554/stream1"},
    {"id": "cam2", "label": "Camera 2", "url": "rtsp://Banana12a:98819881xyz@192.168.43.217:554/stream1"},
]

# ─── Web Dashboard ───────────────────────────────────────────────────────────
WEB_HOST = "0.0.0.0"
WEB_PORT = 5001  # Different port from facial recognition (5000)

# ─── Directories ─────────────────────────────────────────────────────────────
MODELS_DIR = "models"
DETECTIONS_DIR = "detections"           # Saved detection screenshots
DETECTION_LOG_FILE = "coconut_detections.csv"

# ─── YOLOv8 Model ───────────────────────────────────────────────────────────
# After training, place your best.pt model in models/ directory
# Or use the download_model.py script to fetch from Roboflow
YOLO_MODEL_PATH = "models/coconut_best.pt"

# ─── Detection Settings ─────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.40    # Minimum confidence to count as detection
IOU_THRESHOLD = 0.45           # Non-max suppression IoU threshold
PROCESS_EVERY_N_FRAMES = 2    # Analyze every Nth frame (higher = faster, less CPU)
RESIZE_WIDTH = 640             # Resize frame width for inference (640 is YOLOv8 default)

# ─── Detection Classes ──────────────────────────────────────────────────────
# Class names from the coconut dataset
CLASS_NAMES = ["coconut"]

# ─── Screenshot Settings ────────────────────────────────────────────────────
SAVE_DETECTIONS = True         # Save screenshots of detections
SAVE_COOLDOWN_SECONDS = 5     # Minimum time between saved screenshots

# ─── Telegram Alerts (Optional) ─────────────────────────────────────────────
TELEGRAM_ENABLED = False
TELEGRAM_BOT_TOKEN = "YOUR_BOT_TOKEN_HERE"
TELEGRAM_CHAT_ID = "YOUR_CHAT_ID_HERE"

# ─── Roboflow Dataset (for training) ────────────────────────────────────────
# Used by train_model.py to download dataset and train YOLOv8
ROBOFLOW_API_KEY = "YOUR_API_KEY_HERE"      # Get from https://app.roboflow.com/settings/api
ROBOFLOW_WORKSPACE = "fruit-mtdup"
ROBOFLOW_PROJECT = "coconut-a5ecn"
ROBOFLOW_VERSION = 1
