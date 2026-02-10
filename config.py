"""
Configuration — CCTV Face Recognition System
"""

# ─── Cameras ─────────────────────────────────────────────────────────────────
RTSP_URL = "rtsp://Banana12a:98819881xyz@192.168.43.9:554/stream1"

CAMERA_SOURCES = [
    {"id": "cam1", "label": "Camera 1", "url": "rtsp://Banana12a:98819881xyz@192.168.43.9:554/stream1"},
    {"id": "cam2", "label": "Camera 2", "url": "rtsp://Banana12a:98819881xyz@192.168.43.217:554/stream1"},
]

# ─── Web Dashboard ───────────────────────────────────────────────────────────
WEB_HOST = "0.0.0.0"
WEB_PORT = 5000

# ─── Directories ─────────────────────────────────────────────────────────────
KNOWN_FACES_DIR = "known_faces"          # Face images stored here (auto + manual)
DETECTION_LOG_FILE = "detections.csv"    # CSV log of all events

# ─── Recognition ─────────────────────────────────────────────────────────────
PROCESS_EVERY_N_FRAMES = 3     # Analyze every Nth frame (higher = faster)
RESIZE_SCALE = 0.25            # Shrink frame for processing (0.25 = 1/4 size)
MATCH_TOLERANCE = 0.5          # Face match strictness
                               #   0.4 = strict  |  0.5 = balanced  |  0.6 = lenient

# ─── Auto-Enrollment ─────────────────────────────────────────────────────────
AUTO_ENROLL_CONFIDENCE_FRAMES = 5   # Must see face this many times before enrolling
                                     # Prevents enrolling blurry/false faces
                                     # Lower = faster enrollment, more false positives
                                     # Higher = slower enrollment, more reliable

NEW_FACE_COOLDOWN_SECONDS = 10      # Wait this long between enrolling new faces
                                     # Prevents enrolling the same person twice rapidly

# ─── Telegram Alerts (Optional) ──────────────────────────────────────────────
TELEGRAM_ENABLED = False
TELEGRAM_BOT_TOKEN = "YOUR_BOT_TOKEN_HERE"
TELEGRAM_CHAT_ID = "YOUR_CHAT_ID_HERE"
