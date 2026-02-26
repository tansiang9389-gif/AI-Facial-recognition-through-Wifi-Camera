"""
Coconut Maturity Processor - Main processing orchestrator.
Runs in its own thread, round-robins cameras, performs maturity
classification (via Roboflow API or local YOLOv8), annotates frames
with color-coded bounding boxes, and coordinates logging/alerts/tracking.
"""

import time
import threading
from datetime import datetime

import cv2
import numpy as np

from config import (
    CLASS_COLORS, CLASS_LABELS, CLASS_NAMES,
    PROCESS_EVERY_N_FRAMES, RESIZE_WIDTH,
)
from coconut_detector import create_detector, DetectionSaver, init_log, log_detection
from maturity_tracker import MaturityTracker


class CoconutProcessor:
    """Processes camera frames for coconut maturity classification."""

    def __init__(self, cameras):
        self.cameras = cameras
        self.detector = create_detector()
        self.saver = DetectionSaver()
        self.tracker = MaturityTracker()
        self.running = False

        # Per-camera annotated JPEG frames for MJPEG streaming
        self._frames = {}
        self._frame_locks = {}
        for cam_id in cameras:
            self._frames[cam_id] = None
            self._frame_locks[cam_id] = threading.Lock()

        # Per-camera latest stats for the API
        self._stats = {}
        self._stats_lock = threading.Lock()

        # Recent detections list (for the dashboard)
        self._recent_detections = []
        self._recent_lock = threading.Lock()

        init_log()

    def start(self):
        self.running = True
        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False

    def get_annotated_frame(self, cam_id):
        """Return the latest annotated JPEG bytes for a camera."""
        lock = self._frame_locks.get(cam_id)
        if not lock:
            return None
        with lock:
            return self._frames.get(cam_id)

    def get_stats(self):
        """Return current stats dict for all cameras."""
        with self._stats_lock:
            return dict(self._stats)

    def get_recent_detections(self, limit=20):
        """Return recent detections list."""
        with self._recent_lock:
            return list(self._recent_detections[-limit:])

    def get_tracker(self):
        """Access the maturity tracker for trend data."""
        return self.tracker

    def _process_loop(self):
        cam_ids = list(self.cameras.keys())
        cam_idx = 0
        frame_count = 0

        while self.running:
            if not cam_ids:
                time.sleep(1)
                continue

            cam_id = cam_ids[cam_idx % len(cam_ids)]
            cam = self.cameras[cam_id]
            cam_idx += 1
            frame_count += 1

            # Skip frames for performance
            if frame_count % PROCESS_EVERY_N_FRAMES != 0:
                time.sleep(0.01)
                continue

            frame = cam.get_frame()
            if frame is None:
                self._set_no_signal(cam_id, cam.label)
                time.sleep(0.1)
                continue

            orig_h, orig_w = frame.shape[:2]

            # Resize for inference
            scale = RESIZE_WIDTH / orig_w
            resized = cv2.resize(frame, (RESIZE_WIDTH, int(orig_h * scale)))

            # Detect with color validation
            detections = self.detector.detect_with_color_validation(resized)

            # Scale bounding boxes back to original frame size
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                det["bbox"] = (x1 / scale, y1 / scale, x2 / scale, y2 / scale)

            # Count per class
            counts = {cls: 0 for cls in CLASS_NAMES}
            total_conf = 0
            for det in detections:
                cls = det["class_name"]
                if cls in counts:
                    counts[cls] += 1
                total_conf += det["confidence"]

            avg_conf = total_conf / len(detections) if detections else 0
            total = sum(counts.values())

            # Draw annotations on original frame
            annotated = self._draw_annotations(frame, detections, cam.label, counts)

            # Encode to JPEG for streaming
            _, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
            jpeg_bytes = jpeg.tobytes()

            with self._frame_locks[cam_id]:
                self._frames[cam_id] = jpeg_bytes

            # Update stats
            with self._stats_lock:
                self._stats[cam_id] = {
                    "camera": cam.label,
                    "connected": cam.connected,
                    "total_coconuts": total,
                    "premature": counts.get("Premature", 0),
                    "mature": counts.get("Mature", 0),
                    "potential": counts.get("Potential", 0),
                    "avg_confidence": round(avg_conf, 2),
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                }

            # Record in tracker + log + save screenshot
            if total > 0:
                self.tracker.record_detection(cam.label, detections)
                screenshot_path = self.saver.save_if_ready(annotated, cam.label, counts)
                log_detection(cam.label, counts, avg_conf, "detected", screenshot_path or "")

                # Add to recent detections
                det_entry = {
                    "camera": cam.label,
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "premature": counts.get("Premature", 0),
                    "mature": counts.get("Mature", 0),
                    "potential": counts.get("Potential", 0),
                    "total": total,
                    "confidence": round(avg_conf, 2),
                    "screenshot": screenshot_path or "",
                }
                with self._recent_lock:
                    self._recent_detections.append(det_entry)
                    if len(self._recent_detections) > 100:
                        self._recent_detections = self._recent_detections[-100:]

            time.sleep(0.02)

    def _draw_annotations(self, frame, detections, cam_label, counts):
        """Draw color-coded bounding boxes and HUD overlay on the frame."""
        annotated = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            cls = det["class_name"]
            conf = det["confidence"]
            color = CLASS_COLORS.get(cls, (255, 255, 255))
            label_text = CLASS_LABELS.get(cls, cls)

            # Bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Label background
            label_str = f"{label_text} {conf:.0%}"
            if det.get("color_adjusted"):
                label_str += " *"
            (tw, th), _ = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(annotated, label_str, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # HUD overlay - top bar
        h, w = annotated.shape[:2]
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0, annotated)

        # Camera label
        cv2.putText(annotated, cam_label, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Maturity counts in HUD
        total = sum(counts.values())
        x_pos = 10
        y_pos = 50
        for cls_name in CLASS_NAMES:
            cnt = counts.get(cls_name, 0)
            color = CLASS_COLORS.get(cls_name, (255, 255, 255))
            label = CLASS_LABELS.get(cls_name, cls_name)
            text = f"{label}: {cnt}"
            cv2.putText(annotated, text, (x_pos, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            x_pos += 140

        # Total count badge
        cv2.putText(annotated, f"Total: {total}", (x_pos, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Timestamp
        ts = datetime.now().strftime("%H:%M:%S")
        cv2.putText(annotated, ts, (w - 100, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        return annotated

    def _set_no_signal(self, cam_id, cam_label):
        """Generate a 'No Signal' placeholder frame."""
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder, "NO SIGNAL", (180, 230),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 200), 3)
        cv2.putText(placeholder, cam_label, (220, 280),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 2)

        _, jpeg = cv2.imencode(".jpg", placeholder)
        with self._frame_locks[cam_id]:
            self._frames[cam_id] = jpeg.tobytes()

        with self._stats_lock:
            self._stats[cam_id] = {
                "camera": cam_label,
                "connected": False,
                "total_coconuts": 0,
                "premature": 0, "mature": 0, "potential": 0,
                "avg_confidence": 0,
                "timestamp": datetime.now().strftime("%H:%M:%S"),
            }
