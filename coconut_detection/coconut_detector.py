"""
Coconut Maturity Detector - Multi-mode inference engine
Supports 3 inference modes:
  1. "roboflow_hosted"  - Roboflow Serverless API (no local GPU needed)
  2. "roboflow_workflow" - Roboflow Workflow API (detect-and-classify)
  3. "local"            - Local YOLOv8 .pt model file

Also includes:
  - Color histogram secondary classifier
  - Telegram alerting (harvest-readiness)
  - CSV detection logging
  - Detection screenshot saving
"""

import os
import csv
import time
import base64
import threading
from datetime import datetime

import cv2
import numpy as np
import requests as http_requests

from config import (
    INFERENCE_MODE, YOLO_MODEL_PATH, CLASS_NAMES,
    CONFIDENCE_THRESHOLD, IOU_THRESHOLD,
    ROBOFLOW_API_KEY, ROBOFLOW_MODEL_VERSION, ROBOFLOW_PROJECT,
    ROBOFLOW_WORKSPACE, ROBOFLOW_WORKFLOW_ID,
    TELEGRAM_ENABLED, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID,
    DETECTION_LOG_FILE, DETECTIONS_DIR, SAVE_DETECTIONS, SAVE_COOLDOWN_SECONDS,
    COLOR_HISTOGRAM_ENABLED, COLOR_PREMATURE_HUE_RANGE,
    COLOR_MATURE_HUE_RANGE, COLOR_POTENTIAL_HUE_RANGE,
)


# ── Telegram Alerts ──────────────────────────────────────────────────────────

def send_telegram_alert(message, image_path=None):
    """Send text or photo+caption via Telegram Bot API."""
    if not TELEGRAM_ENABLED:
        return
    try:
        if image_path and os.path.exists(image_path):
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
            with open(image_path, "rb") as photo:
                http_requests.post(
                    url,
                    data={"chat_id": TELEGRAM_CHAT_ID, "caption": message},
                    files={"photo": photo},
                    timeout=10,
                )
        else:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            http_requests.post(
                url,
                data={"chat_id": TELEGRAM_CHAT_ID, "text": message},
                timeout=10,
            )
    except Exception as e:
        print(f"[Telegram] Error: {e}")


def send_alert_async(message, image_path=None):
    """Non-blocking Telegram alert."""
    t = threading.Thread(target=send_telegram_alert, args=(message, image_path), daemon=True)
    t.start()


# ── CSV Detection Logger ─────────────────────────────────────────────────────

def init_log():
    """Create the CSV log file with headers if it doesn't exist."""
    if not os.path.exists(DETECTION_LOG_FILE):
        with open(DETECTION_LOG_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "camera",
                "premature_count", "mature_count", "potential_count",
                "total_count", "avg_confidence", "event", "screenshot",
            ])


def log_detection(camera_label, counts, avg_confidence, event="detected", screenshot_path=""):
    """Append a detection record to the CSV log."""
    try:
        with open(DETECTION_LOG_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            total = sum(counts.values())
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                camera_label,
                counts.get("Premature", 0),
                counts.get("Mature", 0),
                counts.get("Potential", 0),
                total,
                f"{avg_confidence:.2f}",
                event,
                screenshot_path,
            ])
    except Exception as e:
        print(f"[Log] Error: {e}")


# ── Color Histogram Classifier ───────────────────────────────────────────────

class ColorHistogramClassifier:
    """Secondary classifier using HSV color analysis of the coconut ROI."""

    @staticmethod
    def classify(frame_bgr, bbox):
        """Classify a coconut ROI by husk color.

        Returns:
            predicted class name or None if uncertain
        """
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = frame_bgr.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 - x1 < 10 or y2 - y1 < 10:
            return None

        roi = frame_bgr[y1:y2, x1:x2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Compute mean hue and saturation (ignore very dark/bright pixels)
        mask = cv2.inRange(hsv, (0, 30, 30), (180, 255, 230))
        if cv2.countNonZero(mask) < 50:
            return None

        mean_hue = cv2.mean(hsv[:, :, 0], mask=mask)[0]
        mean_sat = cv2.mean(hsv[:, :, 1], mask=mask)[0]

        # Classify based on hue ranges
        if COLOR_PREMATURE_HUE_RANGE[0] <= mean_hue <= COLOR_PREMATURE_HUE_RANGE[1]:
            return "Premature"
        elif COLOR_MATURE_HUE_RANGE[0] <= mean_hue <= COLOR_MATURE_HUE_RANGE[1]:
            return "Mature"
        elif COLOR_POTENTIAL_HUE_RANGE[0] <= mean_hue <= COLOR_POTENTIAL_HUE_RANGE[1]:
            return "Potential"

        return None


# ── Roboflow Hosted Inference ─────────────────────────────────────────────────

class RoboflowHostedDetector:
    """Uses Roboflow Serverless Hosted API for inference (no local GPU)."""

    def __init__(self):
        self.api_key = ROBOFLOW_API_KEY
        self.model_id = f"{ROBOFLOW_PROJECT}/{ROBOFLOW_MODEL_VERSION}"
        self.api_url = f"https://serverless.roboflow.com/{self.model_id}"
        self.conf_threshold = CONFIDENCE_THRESHOLD
        self.class_names = CLASS_NAMES
        self.color_classifier = ColorHistogramClassifier() if COLOR_HISTOGRAM_ENABLED else None
        print(f"[Detector] Roboflow Hosted mode: {self.model_id}")

    def detect(self, frame):
        """Send frame to Roboflow API and parse detections."""
        # Encode frame as base64 JPEG
        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        img_b64 = base64.b64encode(buffer).decode("utf-8")

        try:
            resp = http_requests.post(
                self.api_url,
                params={
                    "api_key": self.api_key,
                    "confidence": int(self.conf_threshold * 100),
                },
                data=img_b64,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=10,
            )

            if resp.status_code != 200:
                print(f"[Detector] API error {resp.status_code}: {resp.text[:200]}")
                return []

            data = resp.json()
            return self._parse_response(data, frame)

        except Exception as e:
            print(f"[Detector] Inference error: {e}")
            return []

    def _parse_response(self, data, frame):
        """Parse Roboflow API response into detection dicts."""
        detections = []
        predictions = data.get("predictions", [])
        h, w = frame.shape[:2]

        for pred in predictions:
            # Roboflow returns center coordinates + width/height
            cx = pred.get("x", 0)
            cy = pred.get("y", 0)
            pw = pred.get("width", 0)
            ph = pred.get("height", 0)
            confidence = pred.get("confidence", 0)
            class_name = pred.get("class", "unknown")

            # Convert center format to corner format
            x1 = cx - pw / 2
            y1 = cy - ph / 2
            x2 = cx + pw / 2
            y2 = cy + ph / 2

            if class_name not in CLASS_NAMES:
                # Try to map common variations
                name_lower = class_name.lower()
                if "premature" in name_lower or "pre" in name_lower:
                    class_name = "Premature"
                elif "mature" in name_lower:
                    class_name = "Mature"
                elif "potential" in name_lower:
                    class_name = "Potential"

            class_id = CLASS_NAMES.index(class_name) if class_name in CLASS_NAMES else -1

            detections.append({
                "bbox": (x1, y1, x2, y2),
                "confidence": confidence,
                "class_name": class_name,
                "class_id": class_id,
                "color_adjusted": False,
            })

        return detections

    def detect_with_color_validation(self, frame):
        """Run hosted detection, then optionally refine with color histogram."""
        detections = self.detect(frame)
        if not self.color_classifier:
            return detections

        for det in detections:
            color_pred = self.color_classifier.classify(frame, det["bbox"])
            if color_pred and color_pred != det["class_name"]:
                if det["confidence"] < 0.65:
                    det["class_name"] = color_pred
                    det["class_id"] = CLASS_NAMES.index(color_pred) if color_pred in CLASS_NAMES else -1
                    det["color_adjusted"] = True

        return detections


# ── Roboflow Workflow Inference ───────────────────────────────────────────────

class RoboflowWorkflowDetector:
    """Uses Roboflow Workflow API (detect-and-classify pipeline)."""

    def __init__(self):
        from inference_sdk import InferenceHTTPClient
        self.client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key=ROBOFLOW_API_KEY,
        )
        self.workspace = ROBOFLOW_WORKSPACE
        self.workflow_id = ROBOFLOW_WORKFLOW_ID
        self.class_names = CLASS_NAMES
        self.color_classifier = ColorHistogramClassifier() if COLOR_HISTOGRAM_ENABLED else None
        print(f"[Detector] Roboflow Workflow mode: {self.workspace}/{self.workflow_id}")

    def detect(self, frame):
        """Send frame to Roboflow workflow and parse results."""
        # Save frame to temp file (workflow SDK needs file path or URL)
        temp_path = os.path.join(DETECTIONS_DIR, "_temp_inference.jpg")
        cv2.imwrite(temp_path, frame)

        try:
            result = self.client.run_workflow(
                workspace_name=self.workspace,
                workflow_id=self.workflow_id,
                images={"image": temp_path},
                use_cache=True,
            )

            return self._parse_workflow_result(result, frame)

        except Exception as e:
            print(f"[Detector] Workflow error: {e}")
            return []
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

    def _parse_workflow_result(self, result, frame):
        """Parse workflow result into detection dicts."""
        detections = []
        if not result:
            return detections

        # Workflow results can vary in structure - handle common formats
        for item in result:
            predictions = item.get("predictions", item.get("output", {}).get("predictions", []))
            if isinstance(predictions, dict):
                predictions = predictions.get("predictions", [])

            for pred in (predictions if isinstance(predictions, list) else []):
                cx = pred.get("x", 0)
                cy = pred.get("y", 0)
                pw = pred.get("width", 0)
                ph = pred.get("height", 0)
                confidence = pred.get("confidence", 0)
                class_name = pred.get("class", "unknown")

                x1 = cx - pw / 2
                y1 = cy - ph / 2
                x2 = cx + pw / 2
                y2 = cy + ph / 2

                class_id = CLASS_NAMES.index(class_name) if class_name in CLASS_NAMES else -1

                detections.append({
                    "bbox": (x1, y1, x2, y2),
                    "confidence": confidence,
                    "class_name": class_name,
                    "class_id": class_id,
                    "color_adjusted": False,
                })

        return detections

    def detect_with_color_validation(self, frame):
        """Run workflow detection with optional color refinement."""
        detections = self.detect(frame)
        if not self.color_classifier:
            return detections

        for det in detections:
            color_pred = self.color_classifier.classify(frame, det["bbox"])
            if color_pred and color_pred != det["class_name"]:
                if det["confidence"] < 0.65:
                    det["class_name"] = color_pred
                    det["class_id"] = CLASS_NAMES.index(color_pred) if color_pred in CLASS_NAMES else -1
                    det["color_adjusted"] = True

        return detections


# ── Local YOLOv8 Detector ────────────────────────────────────────────────────

class LocalYOLODetector:
    """Uses a locally trained YOLOv8 .pt model for inference."""

    def __init__(self, model_path=YOLO_MODEL_PATH):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.class_names = CLASS_NAMES
        self.conf_threshold = CONFIDENCE_THRESHOLD
        self.iou_threshold = IOU_THRESHOLD
        self.color_classifier = ColorHistogramClassifier() if COLOR_HISTOGRAM_ENABLED else None
        print(f"[Detector] Local YOLOv8 mode: {model_path}")

    def detect(self, frame):
        """Run local YOLOv8 inference on a frame."""
        results = self.model(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = self.class_names[class_id] if class_id < len(self.class_names) else "unknown"

                detections.append({
                    "bbox": (x1, y1, x2, y2),
                    "confidence": confidence,
                    "class_name": class_name,
                    "class_id": class_id,
                    "color_adjusted": False,
                })

        return detections

    def detect_with_color_validation(self, frame):
        """Run YOLO detection with optional color histogram refinement."""
        detections = self.detect(frame)
        if not self.color_classifier:
            return detections

        for det in detections:
            color_pred = self.color_classifier.classify(frame, det["bbox"])
            if color_pred and color_pred != det["class_name"]:
                if det["confidence"] < 0.65:
                    det["class_name"] = color_pred
                    det["class_id"] = self.class_names.index(color_pred) if color_pred in self.class_names else -1
                    det["color_adjusted"] = True

        return detections


# ── Detector Factory ──────────────────────────────────────────────────────────

def create_detector():
    """Create the appropriate detector based on INFERENCE_MODE config."""
    mode = INFERENCE_MODE.lower()

    if mode == "roboflow_hosted":
        return RoboflowHostedDetector()
    elif mode == "roboflow_workflow":
        return RoboflowWorkflowDetector()
    elif mode == "local":
        return LocalYOLODetector()
    else:
        print(f"[Detector] Unknown inference mode '{mode}', falling back to roboflow_hosted")
        return RoboflowHostedDetector()


# ── Detection Screenshot Saver ────────────────────────────────────────────────

class DetectionSaver:
    """Saves annotated detection screenshots with per-camera cooldown."""

    def __init__(self):
        self._last_save = {}

    def save_if_ready(self, frame, camera_label, counts):
        """Save a screenshot if cooldown has elapsed."""
        if not SAVE_DETECTIONS:
            return None
        total = sum(counts.values())
        if total == 0:
            return None

        now = time.time()
        last = self._last_save.get(camera_label, 0)
        if now - last < SAVE_COOLDOWN_SECONDS:
            return None

        self._last_save[camera_label] = now
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        cam_safe = camera_label.replace(" ", "_")
        filename = (
            f"maturity_{cam_safe}_{ts}"
            f"_pre{counts.get('Premature', 0)}"
            f"_mat{counts.get('Mature', 0)}"
            f"_pot{counts.get('Potential', 0)}.jpg"
        )
        path = os.path.join(DETECTIONS_DIR, filename)

        try:
            cv2.imwrite(path, frame)
            return path
        except Exception as e:
            print(f"[Saver] Error: {e}")
            return None
