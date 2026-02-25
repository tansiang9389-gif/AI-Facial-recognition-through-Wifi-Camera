"""
Coconut Detection Engine — YOLOv8-based coconut detection.
Uses Ultralytics YOLOv8 for real-time object detection.

Supports:
- Real-time coconut detection from RTSP camera feeds
- Confidence-based filtering
- Detection logging and screenshot capture
- Telegram alerts for detections
"""

import cv2
import numpy as np
import os
import csv
import time
import threading
from datetime import datetime

from config import (
    YOLO_MODEL_PATH,
    CONFIDENCE_THRESHOLD,
    IOU_THRESHOLD,
    CLASS_NAMES,
    DETECTIONS_DIR,
    DETECTION_LOG_FILE,
    SAVE_DETECTIONS,
    SAVE_COOLDOWN_SECONDS,
    TELEGRAM_ENABLED,
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_CHAT_ID,
)


# ─── Telegram Alerts ────────────────────────────────────────────────────────

def send_telegram_alert(message, image_path=None):
    if not TELEGRAM_ENABLED:
        return
    try:
        import requests
        if image_path and os.path.exists(image_path):
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
            with open(image_path, "rb") as photo:
                requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "caption": message},
                              files={"photo": photo}, timeout=10)
        else:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": message}, timeout=10)
    except Exception as e:
        print(f"[Telegram Error] {e}")


def send_alert_async(message, image_path=None):
    t = threading.Thread(target=send_telegram_alert, args=(message, image_path))
    t.daemon = True
    t.start()


# ─── Detection Logging ──────────────────────────────────────────────────────

def init_log():
    if not os.path.exists(DETECTION_LOG_FILE):
        with open(DETECTION_LOG_FILE, "w", newline="") as f:
            csv.writer(f).writerow([
                "timestamp", "camera", "num_coconuts",
                "avg_confidence", "event", "screenshot"
            ])


def log_detection(camera_label, num_coconuts, avg_confidence, event="detected", screenshot_path=""):
    with open(DETECTION_LOG_FILE, "a", newline="") as f:
        csv.writer(f).writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            camera_label, num_coconuts, f"{avg_confidence:.2f}",
            event, screenshot_path,
        ])


# ─── Coconut Detector ───────────────────────────────────────────────────────

class CoconutDetector:
    """YOLOv8-based coconut detector.

    Uses Ultralytics YOLO for inference. Supports both custom-trained
    coconut models and fine-tuned models from Roboflow datasets.
    """

    def __init__(self, model_path=YOLO_MODEL_PATH):
        from ultralytics import YOLO

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found: {model_path}\n"
                f"Please either:\n"
                f"  1. Run 'python train_model.py' to train a model on the Roboflow coconut dataset\n"
                f"  2. Run 'python download_model.py' to download a pre-trained model\n"
                f"  3. Place your own YOLOv8 .pt model at: {model_path}"
            )

        self.model = YOLO(model_path)
        self.class_names = CLASS_NAMES
        self.conf_threshold = CONFIDENCE_THRESHOLD
        self.iou_threshold = IOU_THRESHOLD
        print(f"  Coconut detector loaded: {model_path}")
        print(f"  Classes: {self.class_names}")
        print(f"  Confidence threshold: {self.conf_threshold}")

    def detect(self, frame):
        """Run coconut detection on a frame.

        Returns list of detections, each containing:
            - bbox: (x1, y1, x2, y2) bounding box
            - confidence: detection confidence score
            - class_name: detected class name
            - class_id: detected class index
        """
        results = self.model(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())

                # Map class_id to class name
                if class_id < len(self.class_names):
                    class_name = self.class_names[class_id]
                else:
                    class_name = result.names.get(class_id, f"class_{class_id}")

                detections.append({
                    "bbox": (x1, y1, x2, y2),
                    "confidence": confidence,
                    "class_name": class_name,
                    "class_id": class_id,
                })

        return detections


# ─── Detection Screenshot Manager ───────────────────────────────────────────

class DetectionSaver:
    """Manages saving detection screenshots with cooldown."""

    def __init__(self, save_dir=DETECTIONS_DIR):
        self.save_dir = save_dir
        self.last_save_time = 0
        os.makedirs(save_dir, exist_ok=True)

    def save_if_ready(self, frame, camera_label, num_coconuts):
        """Save a detection screenshot if cooldown has elapsed."""
        if not SAVE_DETECTIONS:
            return None

        now = time.time()
        if now - self.last_save_time < SAVE_COOLDOWN_SECONDS:
            return None

        self.last_save_time = now
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"coconut_{camera_label}_{ts}_{num_coconuts}.jpg"
        save_path = os.path.join(self.save_dir, filename)
        cv2.imwrite(save_path, frame)
        return save_path
