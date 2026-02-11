"""
Face Engine — OpenCV DNN-based face detection and recognition.
Uses YuNet (detection) + SFace (recognition), all CPU via OpenCV.

Multi-embedding approach: stores up to MAX_EMBEDDINGS_PER_PERSON embeddings
per person to handle different angles/lighting from RTSP cameras.

Performance: Detection ~13ms + Recognition ~9ms = ~22ms per face (~45 FPS).
"""

import cv2
import numpy as np
import os
import json
import csv
import time
import threading
from datetime import datetime

from config import (
    KNOWN_FACES_DIR,
    DETECTION_LOG_FILE,
    MATCH_TOLERANCE,
    AUTO_ENROLL_CONFIDENCE_FRAMES,
    NEW_FACE_COOLDOWN_SECONDS,
    TELEGRAM_ENABLED,
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_CHAT_ID,
)

FACE_DB_FILE = "face_database.json"
MODELS_DIR = "models"
DETECT_MODEL = os.path.join(MODELS_DIR, "face_detection_yunet_2023mar.onnx")
RECOG_MODEL = os.path.join(MODELS_DIR, "face_recognition_sface_2021dec.onnx")

# ─── Recognition Tuning ─────────────────────────────────────────────────────
COSINE_THRESHOLD = 0.30         # Main match threshold — lowered for RTSP camera quality
ENROLL_THRESHOLD = 0.20         # Threshold used by PendingFaceTracker to group pending faces
MAX_EMBEDDINGS_PER_PERSON = 5   # Store multiple embeddings per person for better matching
LEARN_EVERY_N = 10              # Add a new embedding every N recognitions to adapt over time

# ─── Detection Filtering ────────────────────────────────────────────────────
DETECT_CONFIDENCE = 0.85        # YuNet detection confidence (0.85 = good balance)
MIN_FACE_SIZE = 50              # Minimum face width/height in pixels


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


# ─── Logging ─────────────────────────────────────────────────────────────────

def init_log():
    if not os.path.exists(DETECTION_LOG_FILE):
        with open(DETECTION_LOG_FILE, "w", newline="") as f:
            csv.writer(f).writerow(["timestamp", "person_id", "name", "confidence", "event", "screenshot"])


def log_detection(person_id, name, confidence, event="seen", screenshot_path=""):
    with open(DETECTION_LOG_FILE, "a", newline="") as f:
        csv.writer(f).writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            person_id, name, f"{confidence:.2f}", event, screenshot_path,
        ])


# ─── Face Detector / Recognizer ─────────────────────────────────────────────

class FaceDetector:
    """Wraps OpenCV's YuNet face detector + SFace recognizer.

    Detection: OpenCV FaceDetectorYN (CPU) — ~13ms per frame
    Recognition: OpenCV SFace (CPU) — ~9ms per face
    Applies confidence, size, aspect ratio, and landmark filtering
    to reject false positives (fans, phones, posters, etc.).
    """

    def __init__(self):
        if not os.path.exists(DETECT_MODEL):
            raise FileNotFoundError(f"Detection model not found: {DETECT_MODEL}")
        if not os.path.exists(RECOG_MODEL):
            raise FileNotFoundError(f"Recognition model not found: {RECOG_MODEL}")

        self._detector = cv2.FaceDetectorYN.create(
            DETECT_MODEL, "", (320, 320),
            score_threshold=DETECT_CONFIDENCE,
            nms_threshold=0.3,
            top_k=5000,
        )
        self._recognizer = cv2.FaceRecognizerSF.create(RECOG_MODEL, "")
        print("  Recognition engine: CPU (OpenCV SFace)")

    def detect_reference(self, frame):
        """Detect faces in reference photos with lower threshold (0.5).
        Used when loading known faces from disk — clean images need less strict filtering."""
        h, w = frame.shape[:2]
        ref_detector = cv2.FaceDetectorYN.create(
            DETECT_MODEL, "", (w, h),
            score_threshold=0.5,
            nms_threshold=0.3,
            top_k=5000,
        )
        _, raw_faces = ref_detector.detect(frame)
        if raw_faces is None:
            return []
        return sorted(raw_faces, key=lambda f: float(f[-1]), reverse=True)

    def detect(self, frame):
        """Detect faces with strict filtering. Returns only high-confidence real faces."""
        h, w = frame.shape[:2]
        self._detector.setInputSize((w, h))
        _, raw_faces = self._detector.detect(frame)
        if raw_faces is None:
            return []

        good_faces = []
        for face in raw_faces:
            fw, fh = float(face[2]), float(face[3])

            # Skip faces below minimum size
            if fw < MIN_FACE_SIZE or fh < MIN_FACE_SIZE:
                continue

            # Skip unrealistic aspect ratio
            if fh > 0:
                ratio = fw / fh
                if ratio < 0.4 or ratio > 2.0:
                    continue

            # Validate landmark geometry (eyes, nose, mouth positions)
            if len(face) >= 14:
                rx, ry = float(face[4]), float(face[5])    # right eye
                lx, ly = float(face[6]), float(face[7])    # left eye
                nx, ny = float(face[8]), float(face[9])    # nose
                rmx, rmy = float(face[10]), float(face[11])  # right mouth
                lmx, lmy = float(face[12]), float(face[13])  # left mouth

                eye_dy = abs(ry - ly)
                eye_dx = abs(rx - lx)
                if eye_dx > 0 and eye_dy / eye_dx > 1.0:
                    continue

                eye_mid_y = (ry + ly) / 2
                if ny < eye_mid_y - fh * 0.1:
                    continue

                mouth_mid_y = (rmy + lmy) / 2
                if mouth_mid_y < ny - fh * 0.1:
                    continue

            good_faces.append(face)

        return good_faces

    def get_embedding(self, frame, face):
        """Get 128-d face embedding for a detected face."""
        aligned = self._recognizer.alignCrop(frame, face)
        return self._recognizer.feature(aligned)

    def compare(self, emb1, emb2):
        """Compare two embeddings. Returns cosine similarity score."""
        return self._recognizer.match(emb1, emb2, cv2.FaceRecognizerSF_FR_COSINE)

    def face_bbox(self, face):
        """Extract (left, top, right, bottom) from a face detection result."""
        x, y, w, h = int(face[0]), int(face[1]), int(face[2]), int(face[3])
        return (x, y, x + w, y + h)


# ─── Face Database (Multi-Embedding) ────────────────────────────────────────

class FaceDatabase:
    """Stores multiple embeddings per person for robust matching.
    Periodically learns new embeddings from live detections."""

    def __init__(self, detector, db_file=FACE_DB_FILE, faces_dir=KNOWN_FACES_DIR):
        self.detector = detector
        self.db_file = db_file
        self.faces_dir = faces_dir
        self.persons = {}
        self.person_embeddings = {}
        self.next_id = 1
        self._recognition_count = {}
        os.makedirs(faces_dir, exist_ok=True)
        self._load()

    def _load(self):
        self.person_embeddings = {}

        if os.path.exists(self.db_file):
            with open(self.db_file, "r") as f:
                data = json.load(f)
                self.persons = data.get("persons", {})
                self.next_id = data.get("next_id", 1)

        self._scan_for_new_images()

        for pid, info in sorted(self.persons.items()):
            img_path = os.path.join(self.faces_dir, info["image"])
            if not os.path.exists(img_path):
                continue
            try:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                h, w = img.shape[:2]
                if max(h, w) > 800:
                    scale = 800 / max(h, w)
                    img = cv2.resize(img, None, fx=scale, fy=scale)

                # Use lower threshold for reference photos (known-good images)
                faces = self.detector.detect_reference(img)
                if len(faces) > 0:
                    emb = self.detector.get_embedding(img, faces[0])
                    self.person_embeddings[pid] = [emb]
                else:
                    print(f"  Warning: No face detected in {img_path}")
            except Exception as e:
                print(f"  Warning: Could not load {img_path}: {e}")

        total_emb = sum(len(v) for v in self.person_embeddings.values())
        print(f"  Database: {len(self.person_embeddings)} person(s), {total_emb} embedding(s)")

    def _scan_for_new_images(self):
        valid_ext = (".jpg", ".jpeg", ".png", ".bmp")
        existing_images = {info["image"] for info in self.persons.values()}

        for filename in sorted(os.listdir(self.faces_dir)):
            if not filename.lower().endswith(valid_ext):
                continue
            if filename in existing_images:
                continue
            if "_full." in filename:
                continue
            name = os.path.splitext(filename)[0].replace("_", " ").title()
            pid = f"P{self.next_id:04d}"
            self.next_id += 1
            self.persons[pid] = {
                "name": name,
                "image": filename,
                "first_seen": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "last_seen": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            print(f"  Found new image: {filename} -> {pid} ({name})")
        self._save()

    def _save(self):
        with open(self.db_file, "w") as f:
            json.dump({"persons": self.persons, "next_id": self.next_id}, f, indent=2)

    def match(self, embedding):
        """Match embedding against all stored embeddings. Returns (pid, name, score)."""
        if not self.person_embeddings:
            return None, None, 0.0

        best_score = -1.0
        best_pid = None

        for pid, emb_list in self.person_embeddings.items():
            for stored_emb in emb_list:
                score = self.detector.compare(embedding, stored_emb)
                if score > best_score:
                    best_score = score
                    best_pid = pid

        if best_score >= COSINE_THRESHOLD and best_pid in self.persons:
            name = self.persons[best_pid]["name"]
            self.persons[best_pid]["last_seen"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self._maybe_learn(best_pid, embedding)
            return best_pid, name, best_score

        return None, None, best_score

    def _maybe_learn(self, pid, embedding):
        """Add a new embedding periodically to adapt to different angles/lighting."""
        count = self._recognition_count.get(pid, 0) + 1
        self._recognition_count[pid] = count

        emb_list = self.person_embeddings.get(pid, [])
        if len(emb_list) >= MAX_EMBEDDINGS_PER_PERSON:
            return

        if count % LEARN_EVERY_N == 0:
            for existing in emb_list:
                if self.detector.compare(embedding, existing) > 0.8:
                    return
            emb_list.append(embedding)
            self.person_embeddings[pid] = emb_list
            print(f"  [Learn] {self.persons[pid]['name']} ({pid}): now {len(emb_list)} embeddings")

    def enroll_new(self, frame, face_bbox, embedding):
        """Enroll a new person. Returns (pid, name, save_path)."""
        pid = f"P{self.next_id:04d}"
        self.next_id += 1
        name = f"Person {int(pid[1:])}"
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{pid}_{ts}.jpg"

        left, top, right, bottom = face_bbox
        h, w = frame.shape[:2]
        pad_h = int((bottom - top) * 0.4)
        pad_w = int((right - left) * 0.4)
        crop = frame[max(0, top - pad_h):min(h, bottom + pad_h),
                      max(0, left - pad_w):min(w, right + pad_w)]

        save_path = os.path.join(self.faces_dir, filename)
        cv2.imwrite(save_path, crop)

        full_path = os.path.join(self.faces_dir, f"{pid}_{ts}_full.jpg")
        cv2.imwrite(full_path, frame)

        self.persons[pid] = {
            "name": name,
            "image": filename,
            "first_seen": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "last_seen": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.person_embeddings[pid] = [embedding]
        self._save()
        return pid, name, save_path

    def rename(self, pid, new_name):
        if pid in self.persons:
            old_name = self.persons[pid]["name"]
            self.persons[pid]["name"] = new_name
            self._save()
            return old_name
        return None

    def list_persons(self):
        return dict(sorted(self.persons.items()))

    def total(self):
        return len(self.person_embeddings)


# ─── Pending Face Tracker ───────────────────────────────────────────────────

class PendingFaceTracker:
    """Tracks unrecognized faces. Only enrolls after N consistent sightings."""

    def __init__(self, detector, required_frames=AUTO_ENROLL_CONFIDENCE_FRAMES):
        self.detector = detector
        self.required_frames = required_frames
        self.pending = []
        self.cooldown_until = 0

    def track(self, embedding, frame, face_bbox):
        now = time.time()
        if now < self.cooldown_until:
            return None

        best_idx = -1
        best_score = -1.0
        for i, entry in enumerate(self.pending):
            score = self.detector.compare(entry["embedding"], embedding)
            if score > best_score and score >= ENROLL_THRESHOLD:
                best_score = score
                best_idx = i

        if best_idx >= 0:
            entry = self.pending[best_idx]
            entry["count"] += 1
            entry["last_seen"] = now
            entry["best_frame"] = frame.copy()
            entry["best_bbox"] = face_bbox
            if np.linalg.norm(embedding) > np.linalg.norm(entry["embedding"]):
                entry["embedding"] = embedding

            if entry["count"] >= self.required_frames:
                self.pending.pop(best_idx)
                self.cooldown_until = now + NEW_FACE_COOLDOWN_SECONDS
                return entry
        else:
            self.pending.append({
                "embedding": embedding,
                "count": 1,
                "last_seen": now,
                "best_frame": frame.copy(),
                "best_bbox": face_bbox,
            })

        self.pending = [e for e in self.pending if now - e["last_seen"] < 10]
        return None
