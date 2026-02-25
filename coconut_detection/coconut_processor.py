"""
Coconut Processing Thread
Alternates between cameras, runs YOLOv8 coconut detection.
Annotates frames with bounding boxes and coconut count.
"""

import cv2
import numpy as np
import threading
import time
from datetime import datetime

from coconut_detector import (
    CoconutDetector, DetectionSaver,
    send_alert_async, log_detection, init_log,
)
from config import PROCESS_EVERY_N_FRAMES, RESIZE_WIDTH


# ─── Color palette for drawing ───────────────────────────────────────────────
COCONUT_COLOR = (0, 200, 0)        # Green for detected coconuts
HUD_COLOR = (255, 255, 255)        # White for HUD text
COUNT_COLOR = (0, 255, 255)        # Yellow for coconut count


class CoconutProcessor:
    def __init__(self, cameras):
        """cameras: dict of {cam_id: CameraStream}"""
        self.cameras = cameras

        print("  Loading YOLOv8 coconut detection model...")
        self.detector = CoconutDetector()
        self.saver = DetectionSaver()
        init_log()

        # Annotated frames for MJPEG streaming (cam_id -> jpeg bytes)
        self.annotated_frames = {}
        self.frame_locks = {cam_id: threading.Lock() for cam_id in cameras}

        # Detection stats tracking
        self.detection_stats = {}   # cam_id -> {count, last_seen, ...}
        self.stats_lock = threading.Lock()
        self.total_detections = 0

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

    def get_stats(self):
        with self.stats_lock:
            now = time.time()
            return {
                "total_detections": self.total_detections,
                "cameras_connected": sum(1 for c in self.cameras.values() if c.connected),
                "cameras_total": len(self.cameras),
                "camera_stats": dict(self.detection_stats),
            }

    def get_detection_history(self):
        with self.stats_lock:
            return dict(self.detection_stats)

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

            # Skip frames for performance
            if frame_count % PROCESS_EVERY_N_FRAMES != 0:
                # Still encode the frame for streaming (without detection)
                _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                with self.frame_locks[cam_id]:
                    self.annotated_frames[cam_id] = buf.tobytes()
                time.sleep(0.01)
                continue

            # Resize for inference
            h, w = frame.shape[:2]
            if w > RESIZE_WIDTH:
                scale = RESIZE_WIDTH / w
                inference_frame = cv2.resize(frame, None, fx=scale, fy=scale)
            else:
                inference_frame = frame
                scale = 1.0

            # Run YOLOv8 coconut detection
            detections = self.detector.detect(inference_frame)

            # Scale detections back to original frame size
            if scale != 1.0:
                for det in detections:
                    x1, y1, x2, y2 = det["bbox"]
                    det["bbox"] = (
                        int(x1 / scale), int(y1 / scale),
                        int(x2 / scale), int(y2 / scale),
                    )

            num_coconuts = len(detections)

            # Periodic status log
            if frame_count % 50 == 1:
                print(f"[Processor] {cam_id}: frame #{frame_count}, coconuts: {num_coconuts}")

            # Draw detections on frame
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                conf = det["confidence"]
                label = f"{det['class_name']} {conf:.0%}"

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), COCONUT_COLOR, 2)

                # Draw label background and text
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y2), (x1 + tw + 10, y2 + th + 14),
                              COCONUT_COLOR, cv2.FILLED)
                cv2.putText(frame, label, (x1 + 5, y2 + th + 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Draw coconut count badge
            count_text = f"Coconuts: {num_coconuts}"
            cv2.putText(frame, count_text, (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, COUNT_COLOR, 2)

            # Draw HUD
            status = "LIVE" if cam.connected else "OFFLINE"
            hud = f"{cam.label} | {status} | Coconuts: {num_coconuts}"
            cv2.putText(frame, hud, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, HUD_COLOR, 2)

            # Update stats
            now = time.time()
            if num_coconuts > 0:
                self.total_detections += num_coconuts
                avg_conf = np.mean([d["confidence"] for d in detections])

                with self.stats_lock:
                    self.detection_stats[cam_id] = {
                        "camera": cam.label,
                        "count": num_coconuts,
                        "avg_confidence": round(float(avg_conf), 2),
                        "last_seen": datetime.now().strftime("%H:%M:%S"),
                        "last_seen_ts": now,
                    }

                # Log detection
                log_detection(cam.label, num_coconuts, avg_conf)

                # Save screenshot
                save_path = self.saver.save_if_ready(frame, cam_id, num_coconuts)
                if save_path:
                    send_alert_async(
                        f"Coconut detected: {num_coconuts} coconut(s)\n"
                        f"{cam.label}\n"
                        f"{datetime.now():%Y-%m-%d %H:%M:%S}",
                        save_path,
                    )

            # Encode to JPEG for MJPEG stream
            _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            with self.frame_locks[cam_id]:
                self.annotated_frames[cam_id] = buf.tobytes()

            # Small sleep to not spin CPU at 100%
            time.sleep(0.03)
