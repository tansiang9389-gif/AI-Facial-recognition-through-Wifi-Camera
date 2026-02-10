"""
Face Engine — OpenCV DNN-based face detection + ONNX Runtime GPU recognition.
Uses YuNet (detection, CPU) + SFace via ONNX Runtime (recognition, GPU/CUDA).

Multi-embedding approach: stores up to MAX_EMBEDDINGS_PER_PERSON embeddings
per person to handle different angles/lighting from RTSP cameras.

GPU Acceleration:
- Detection runs on CPU via OpenCV FaceDetectorYN (already fast, ~13ms)
- Recognition runs on GPU via ONNX Runtime CUDAExecutionProvider (~2-3ms)
- Falls back to CPU automatically if CUDA is unavailable
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
# Cosine similarity thresholds (higher = more similar, range roughly -1 to 1)
COSINE_THRESHOLD = 0.30         # Main match threshold — lowered for RTSP camera quality
ENROLL_THRESHOLD = 0.20         # Threshold used by PendingFaceTracker to group pending faces
MAX_EMBEDDINGS_PER_PERSON = 5   # Store multiple embeddings per person for better matching
LEARN_EVERY_N = 10              # Add a new embedding every N recognitions to adapt over time

# ─── Detection Filtering ────────────────────────────────────────────────────
DETECT_CONFIDENCE = 0.85        # YuNet detection confidence threshold (0.0-1.0)
                                # 0.5 = too many false positives (fans, posters, phones)
                                # 0.85 = good balance — rejects most non-faces
                                # 0.95 = very strict — may miss some real faces at bad angles
MIN_FACE_SIZE = 50              # Minimum face width/height in pixels to consider
                                # Rejects tiny false detections from distant objects


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


# ─── ONNX Runtime GPU Session ───────────────────────────────────────────────

def _create_ort_session(model_path):
    """Create an ONNX Runtime session with GPU (CUDA) if available, else CPU."""
    try:
        import onnxruntime as ort

        # Try CUDA first, then TensorRT, then CPU
        providers_to_try = []

        available = ort.get_available_providers()

        if "CUDAExecutionProvider" in available:
            providers_to_try.append(("CUDAExecutionProvider", {
                "device_id": 0,
                "arena_extend_strategy": "kSameAsRequested",
                "cudnn_conv_algo_search": "DEFAULT",
            }))

        providers_to_try.append(("CPUExecutionProvider", {}))

        session = ort.InferenceSession(
            model_path,
            providers=[p[0] for p in providers_to_try],
            # Provider options
        )

        # Set provider options properly
        session = ort.InferenceSession(
            model_path,
            providers=[(p[0], p[1]) for p in providers_to_try],
        )

        active_provider = session.get_providers()[0]
        using_gpu = "CUDA" in active_provider or "Tensorrt" in active_provider
        return session, using_gpu

    except ImportError:
        return None, False
    except Exception as e:
        print(f"  [ONNX Runtime] Warning: {e}")
        return None, False


# ─── Face Detector / Recognizer Wrapper ─────────────────────────────────────

class FaceDetector:
    """Wraps OpenCV's YuNet face detector + SFace recognizer (GPU-accelerated).

    Detection: OpenCV FaceDetectorYN (CPU) — fast, built-in post-processing
    Recognition: ONNX Runtime CUDAExecutionProvider (GPU) — faster embeddings
    Fallback: OpenCV SFace (CPU) if ONNX Runtime unavailable
    """

    def __init__(self):
        if not os.path.exists(DETECT_MODEL):
            raise FileNotFoundError(f"Detection model not found: {DETECT_MODEL}")
        if not os.path.exists(RECOG_MODEL):
            raise FileNotFoundError(f"Recognition model not found: {RECOG_MODEL}")

        # ── Detection: OpenCV YuNet (CPU) ──
        self._detector = cv2.FaceDetectorYN.create(
            DETECT_MODEL, "", (320, 320),
            score_threshold=DETECT_CONFIDENCE,
            nms_threshold=0.3,
            top_k=5000,
        )

        # ── Recognition: OpenCV SFace for alignment (always needed) ──
        self._recognizer = cv2.FaceRecognizerSF.create(RECOG_MODEL, "")

        # ── Recognition: ONNX Runtime GPU for fast embedding ──
        self._ort_session = None
        self._ort_input_name = None
        self._ort_output_name = None
        self._using_gpu = False

        ort_session, using_gpu = _create_ort_session(RECOG_MODEL)
        if ort_session is not None:
            self._ort_session = ort_session
            self._ort_input_name = ort_session.get_inputs()[0].name
            self._ort_output_name = ort_session.get_outputs()[0].name
            self._using_gpu = using_gpu

            provider = "GPU (CUDA)" if using_gpu else "CPU (ONNX Runtime)"
            print(f"  Recognition engine: {provider}")
            print(f"    Provider: {ort_session.get_providers()[0]}")
            print(f"    Input: {self._ort_input_name} {ort_session.get_inputs()[0].shape}")
            print(f"    Output: {self._ort_output_name} {ort_session.get_outputs()[0].shape}")
        else:
            print("  Recognition engine: CPU (OpenCV SFace)")

    @property
    def gpu_enabled(self):
        return self._using_gpu

    def detect(self, frame):
        """Detect faces with strict filtering. Returns only high-confidence real faces."""
        h, w = frame.shape[:2]
        self._detector.setInputSize((w, h))
        _, raw_faces = self._detector.detect(frame)
        if raw_faces is None:
            return []

        # Filter out false positives
        good_faces = []
        for face in raw_faces:
            fw, fh = float(face[2]), float(face[3])
            conf = float(face[-1])

            # 1. Skip faces below minimum size
            if fw < MIN_FACE_SIZE or fh < MIN_FACE_SIZE:
                continue

            # 2. Skip faces with unrealistic aspect ratio
            #    Real faces are roughly 0.7-1.4 width/height ratio
            if fh > 0:
                ratio = fw / fh
                if ratio < 0.4 or ratio > 2.0:
                    continue

            # 3. Validate landmark geometry (YuNet outputs 5 landmarks:
            #    right_eye, left_eye, nose_tip, right_mouth, left_mouth)
            #    Indices: [4,5]=right_eye, [6,7]=left_eye, [8,9]=nose, [10,11]=r_mouth, [12,13]=l_mouth
            if len(face) >= 14:
                rx, ry = float(face[4]), float(face[5])    # right eye
                lx, ly = float(face[6]), float(face[7])    # left eye
                nx, ny = float(face[8]), float(face[9])    # nose
                rmx, rmy = float(face[10]), float(face[11])  # right mouth
                lmx, lmy = float(face[12]), float(face[13])  # left mouth

                # Eyes should be roughly horizontal (not wildly offset vertically)
                eye_dy = abs(ry - ly)
                eye_dx = abs(rx - lx)
                if eye_dx > 0 and eye_dy / eye_dx > 1.0:
                    continue  # Eyes more vertical than horizontal — not a real face

                # Nose should be below eyes
                eye_mid_y = (ry + ly) / 2
                if ny < eye_mid_y - fh * 0.1:
                    continue  # Nose above eyes — not a real face

                # Mouth should be below nose
                mouth_mid_y = (rmy + lmy) / 2
                if mouth_mid_y < ny - fh * 0.1:
                    continue  # Mouth above nose — not a real face

            good_faces.append(face)

        return good_faces

    def get_embedding(self, frame, face):
        """Get 128-d face embedding for a detected face.
        Uses GPU (ONNX Runtime CUDA) if available, else CPU (OpenCV SFace).
        """
        # Step 1: Align and crop face using OpenCV (always CPU — fast)
        aligned = self._recognizer.alignCrop(frame, face)

        # Step 2: Get embedding — GPU or CPU path
        if self._ort_session is not None:
            return self._get_embedding_ort(aligned)
        else:
            return self._recognizer.feature(aligned)

    def _get_embedding_ort(self, aligned_face):
        """Run SFace recognition model on GPU via ONNX Runtime.
        Input: BGR aligned face image (112x112x3)
        Output: 128-d L2-normalized embedding
        """
        # SFace expects: [1, 3, 112, 112] float32, BGR, normalized to [0,1]
        img = cv2.resize(aligned_face, (112, 112))
        img = img.astype(np.float32) / 255.0
        # HWC -> CHW -> NCHW
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)

        # Run inference on GPU
        outputs = self._ort_session.run(
            [self._ort_output_name],
            {self._ort_input_name: img}
        )

        embedding = outputs[0].flatten()
        # L2 normalize (SFace embeddings should be unit vectors for cosine similarity)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding.reshape(1, -1).astype(np.float32)

    def compare(self, emb1, emb2):
        """Compare two embeddings. Returns cosine similarity score (higher = more similar).
        Works with both OpenCV SFace and ONNX Runtime embeddings.
        """
        if self._ort_session is not None:
            # Numpy cosine similarity for ONNX Runtime embeddings
            e1 = emb1.flatten()
            e2 = emb2.flatten()
            dot = np.dot(e1, e2)
            n1 = np.linalg.norm(e1)
            n2 = np.linalg.norm(e2)
            if n1 > 0 and n2 > 0:
                return float(dot / (n1 * n2))
            return 0.0
        else:
            # OpenCV matcher for OpenCV SFace embeddings
            return self._recognizer.match(emb1, emb2, cv2.FaceRecognizerSF_FR_COSINE)

    def face_bbox(self, face):
        """Extract (left, top, right, bottom) from a face detection result."""
        x, y, w, h = int(face[0]), int(face[1]), int(face[2]), int(face[3])
        return (x, y, x + w, y + h)


# ─── Face Database (Multi-Embedding) ────────────────────────────────────────

class FaceDatabase:
    """
    Stores multiple embeddings per person for robust matching.
    When matching, checks against ALL embeddings and returns the best match.
    Periodically learns new embeddings from live detections.
    """

    def __init__(self, detector, db_file=FACE_DB_FILE, faces_dir=KNOWN_FACES_DIR):
        self.detector = detector
        self.db_file = db_file
        self.faces_dir = faces_dir
        self.persons = {}
        # pid -> list of embeddings
        self.person_embeddings = {}
        self.next_id = 1
        # Track recognition count per person for adaptive learning
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
                # Resize large images for detection
                h, w = img.shape[:2]
                if max(h, w) > 800:
                    scale = 800 / max(h, w)
                    img = cv2.resize(img, None, fx=scale, fy=scale)

                faces = self.detector.detect(img)
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
        """
        Match an embedding against ALL stored embeddings for ALL persons.
        Returns (pid, name, best_score) or (None, None, best_score).
        Also does adaptive learning — periodically stores new embeddings.
        """
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

            # Adaptive learning: occasionally store new embedding for this person
            self._maybe_learn(best_pid, embedding)

            return best_pid, name, best_score

        return None, None, best_score

    def _maybe_learn(self, pid, embedding):
        """Add a new embedding for a person if we haven't stored too many yet."""
        count = self._recognition_count.get(pid, 0) + 1
        self._recognition_count[pid] = count

        emb_list = self.person_embeddings.get(pid, [])
        if len(emb_list) >= MAX_EMBEDDINGS_PER_PERSON:
            return

        # Learn a new embedding every N recognitions
        if count % LEARN_EVERY_N == 0:
            # Check this embedding is different enough from existing ones (not a duplicate)
            for existing in emb_list:
                if self.detector.compare(embedding, existing) > 0.8:
                    return  # Too similar to an existing one, skip

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

        # Crop face with padding
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
            # Keep the best quality embedding (highest norm tends to be better)
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

        # Clean stale entries
        self.pending = [e for e in self.pending if now - e["last_seen"] < 10]
        return None
