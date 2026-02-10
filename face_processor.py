"""
Face Processing Thread
Alternates between cameras, runs face detection/recognition using OpenCV DNN.
No dlib dependency — uses YuNet + SFace models.
"""

import cv2
import numpy as np
import threading
import time
from datetime import datetime

from face_engine import FaceDetector, FaceDatabase, PendingFaceTracker, send_alert_async, log_detection, init_log


class FaceProcessor:
    def __init__(self, cameras):
        """cameras: dict of {cam_id: CameraStream}"""
        self.cameras = cameras

        print("  Loading OpenCV face models...")
        self.detector = FaceDetector()
        self.db = FaceDatabase(self.detector)
        self.tracker = PendingFaceTracker(self.detector)
        init_log()

        # Annotated frames for MJPEG streaming (cam_id -> jpeg bytes)
        self.annotated_frames = {}
        self.frame_locks = {cam_id: threading.Lock() for cam_id in cameras}

        # Person visibility tracking
        self.person_status = {}  # pid -> {name, camera, last_seen, active}
        self.status_lock = threading.Lock()

        self.running = False
        self._thread = None

    def start(self):
        self.running = True
        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False

    def get_annotated_frame(self, cam_id):
        lock = self.frame_locks.get(cam_id)
        if not lock:
            return None
        with lock:
            return self.annotated_frames.get(cam_id)

    def get_persons(self):
        with self.status_lock:
            return dict(self.person_status)

    def get_stats(self):
        with self.status_lock:
            now = time.time()
            active = sum(1 for p in self.person_status.values() if now - p["last_seen_ts"] < 10)
            return {
                "total_persons": self.db.total(),
                "active_now": active,
                "cameras_connected": sum(1 for c in self.cameras.values() if c.connected),
                "cameras_total": len(self.cameras),
            }

    def get_db_persons(self):
        return self.db.list_persons()

    def _process_loop(self):
        cam_ids = list(self.cameras.keys())
        cam_idx = 0
        frame_count = 0

        while self.running:
            cam_id = cam_ids[cam_idx % len(cam_ids)]
            cam = self.cameras[cam_id]
            cam_idx += 1

            frame = cam.get_frame()
            if frame is None:
                # No frame yet — generate a "no signal" placeholder
                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(placeholder, f"{cam.label}: No Signal", (120, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                _, buf = cv2.imencode('.jpg', placeholder, [cv2.IMWRITE_JPEG_QUALITY, 70])
                with self.frame_locks[cam_id]:
                    self.annotated_frames[cam_id] = buf.tobytes()
                time.sleep(0.5)
                continue

            frame_count += 1

            # Detect faces using OpenCV YuNet (works on BGR directly, no RGB conversion needed)
            faces = self.detector.detect(frame)

            # Periodic status log
            if frame_count % 50 == 1:
                print(f"[Processor] {cam_id}: frame #{frame_count}, faces: {len(faces)}, db: {self.db.total()}")

            now = time.time()

            for face in faces:
                left, top, right, bottom = self.detector.face_bbox(face)
                confidence_detect = float(face[-1])

                # Get face embedding
                try:
                    embedding = self.detector.get_embedding(frame, face)
                except Exception:
                    continue

                # Try to match against known faces
                pid, name, score = self.db.match(embedding)

                if pid:
                    # Known person — GREEN box
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 200, 0), 2)
                    label = f"{name} ({score:.0%})"
                    self._draw_label(frame, label, left, top, bottom, (0, 200, 0))
                    log_detection(pid, name, score, "seen")

                    with self.status_lock:
                        self.person_status[pid] = {
                            "name": name,
                            "camera": cam.label,
                            "camera_id": cam_id,
                            "last_seen": datetime.now().strftime("%H:%M:%S"),
                            "last_seen_ts": now,
                            "confidence": round(score, 2),
                        }
                else:
                    # Unknown — try to enroll
                    bbox = (left, top, right, bottom)
                    ready = self.tracker.track(embedding, frame, bbox)

                    if ready:
                        new_pid, new_name, save_path = self.db.enroll_new(
                            ready["best_frame"], ready["best_bbox"], ready["embedding"]
                        )
                        ts = datetime.now().strftime("%H:%M:%S")
                        print(f"[{ts}] NEW PERSON: {new_name} ({new_pid}) on {cam.label}")
                        log_detection(new_pid, new_name, score, "enrolled", save_path)

                        send_alert_async(
                            f"New person: {new_name}\n{cam.label}\n{datetime.now():%Y-%m-%d %H:%M:%S}",
                            save_path,
                        )

                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 255), 2)
                        self._draw_label(frame, f"NEW: {new_name}", left, top, bottom, (0, 255, 255))

                        with self.status_lock:
                            self.person_status[new_pid] = {
                                "name": new_name,
                                "camera": cam.label,
                                "camera_id": cam_id,
                                "last_seen": datetime.now().strftime("%H:%M:%S"),
                                "last_seen_ts": now,
                                "confidence": 0,
                            }
                    else:
                        # Scanning — ORANGE box
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 165, 255), 2)
                        self._draw_label(frame, "Scanning...", left, top, bottom, (0, 165, 255))

            # Draw HUD on frame
            status = "LIVE" if cam.connected else "OFFLINE"
            hud = f"{cam.label} | {status} | DB:{self.db.total()} | Faces:{len(faces)}"
            cv2.putText(frame, hud, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Encode to JPEG for MJPEG stream
            _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            with self.frame_locks[cam_id]:
                self.annotated_frames[cam_id] = buf.tobytes()

            # Small sleep to not spin CPU at 100%
            time.sleep(0.03)

    def _draw_label(self, frame, label, left, top, bottom, color):
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (left, bottom), (left + w + 10, bottom + h + 14), color, cv2.FILLED)
        cv2.putText(frame, label, (left + 5, bottom + h + 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
