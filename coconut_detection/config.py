"""
Coconut Maturity Classification System - Configuration
Uses Roboflow "Coconut Maturity Detection" dataset with 3 classes:
  Premature, Mature, Potential
Supports both Roboflow Hosted Inference API and local YOLOv8 model.
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --------------- Camera Sources ---------------
CAMERA_SOURCES = [
    {
        "id": "cam1",
        "label": "Camera 1",
        "url": "rtsp://Banana12a:98819881xyz@192.168.43.9:554/stream1",
    },
    {
        "id": "cam2",
        "label": "Camera 2",
        "url": "rtsp://Banana12a:98819881xyz@192.168.43.217:554/stream1",
    },
]

# --------------- Web Server ---------------
WEB_HOST = "0.0.0.0"
WEB_PORT = 5001

# --------------- Directories ---------------
MODELS_DIR = os.path.join(BASE_DIR, "models")
DETECTIONS_DIR = os.path.join(BASE_DIR, "detections")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

# --------------- Roboflow API & Inference ---------------
ROBOFLOW_API_KEY = "UMAGXEzJ8rAmf19vMs5F"
ROBOFLOW_WORKSPACE = "coconut-maturity-detection"
ROBOFLOW_PROJECT = "coconut-maturity-detection"
ROBOFLOW_VERSION = 7
ROBOFLOW_MODEL_VERSION = 6  # Version with trained model for hosted inference
ROBOFLOW_WORKFLOW_ID = "detect-and-classify"  # Workflow name on Roboflow

# Inference mode: "roboflow_hosted" uses Roboflow serverless API
#                 "roboflow_workflow" uses Roboflow workflow API
#                 "local" uses a locally trained YOLOv8 .pt file
INFERENCE_MODE = "roboflow_hosted"

# Local YOLOv8 model path (used when INFERENCE_MODE = "local")
YOLO_MODEL_PATH = os.path.join(MODELS_DIR, "coconut_maturity_best.pt")

# --------------- Dataset Classes ---------------
# From: https://universe.roboflow.com/coconut-maturity-detection/coconut-maturity-detection
# 3 classes: Premature (6462), Mature (3649), Potential (4103)
CLASS_NAMES = ["Premature", "Mature", "Potential"]

# Per-class display colors (BGR for OpenCV)
CLASS_COLORS = {
    "Premature": (0, 200, 0),       # green  - young/premature coconut
    "Mature": (0, 140, 255),        # orange - mature coconut, ready to harvest
    "Potential": (0, 215, 255),     # yellow - potentially mature, transitional
}

# Per-class display labels for the UI
CLASS_LABELS = {
    "Premature": "Premature",
    "Mature": "Mature",
    "Potential": "Potential",
}

# Per-class emojis for Telegram messages
CLASS_EMOJI = {
    "Premature": "\U0001F7E2",  # green circle
    "Mature": "\U0001F7E0",     # orange circle
    "Potential": "\U0001F7E1",  # yellow circle
}

# --------------- Detection Thresholds ---------------
CONFIDENCE_THRESHOLD = 0.40
IOU_THRESHOLD = 0.45
PROCESS_EVERY_N_FRAMES = 2
RESIZE_WIDTH = 640

# --------------- Color Histogram Classifier ---------------
# Secondary classifier to validate/adjust predictions using HSV color
COLOR_HISTOGRAM_ENABLED = True
# Green husk (premature): H 25-85
COLOR_PREMATURE_HUE_RANGE = (25, 85)
# Brown husk (mature): H 5-25
COLOR_MATURE_HUE_RANGE = (5, 25)
# Transitional (potential): H 15-35 (greenish-brown)
COLOR_POTENTIAL_HUE_RANGE = (15, 35)

# --------------- Detection Saving ---------------
SAVE_DETECTIONS = True
SAVE_COOLDOWN_SECONDS = 5

# --------------- Logging ---------------
DETECTION_LOG_FILE = os.path.join(BASE_DIR, "coconut_maturity_log.csv")

# --------------- Telegram Alerts ---------------
TELEGRAM_ENABLED = False
TELEGRAM_BOT_TOKEN = "YOUR_BOT_TOKEN_HERE"
TELEGRAM_CHAT_ID = "YOUR_CHAT_ID_HERE"

# Harvest-readiness alert: triggers when Mature coconut count
# exceeds this threshold in a single camera frame
HARVEST_ALERT_THRESHOLD = 5
HARVEST_ALERT_COOLDOWN_MINUTES = 30

# --------------- Daily Reports ---------------
DAILY_REPORT_ENABLED = True
DAILY_REPORT_HOUR = 18  # 6 PM local time

# Ensure directories exist
for d in [MODELS_DIR, DETECTIONS_DIR, REPORTS_DIR]:
    os.makedirs(d, exist_ok=True)
