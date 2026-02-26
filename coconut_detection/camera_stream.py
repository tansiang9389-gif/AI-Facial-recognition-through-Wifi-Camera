"""
Threaded RTSP/WiFi camera reader with auto-reconnection.
Each camera runs in its own daemon thread, providing thread-safe frame access.
"""

import threading
import time
import cv2


class CameraStream:
    def __init__(self, cam_id, url, label="Camera"):
        self.cam_id = cam_id
        self.url = url
        self.label = label
        self.frame = None
        self.lock = threading.Lock()
        self.running = False
        self.connected = False

    def start(self):
        self.running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False

    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def _read_loop(self):
        while self.running:
            cap = cv2.VideoCapture(self.url)
            if not cap.isOpened():
                print(f"[{self.label}] Cannot connect to {self.url}, retrying in 5s...")
                self.connected = False
                time.sleep(5)
                continue

            self.connected = True
            print(f"[{self.label}] Connected to {self.url}")
            fail_count = 0

            while self.running:
                ret, frame = cap.read()
                if not ret:
                    fail_count += 1
                    if fail_count > 30:
                        print(f"[{self.label}] Too many read failures, reconnecting...")
                        break
                    time.sleep(0.01)
                    continue

                fail_count = 0
                with self.lock:
                    self.frame = frame

            cap.release()
            self.connected = False
            if self.running:
                print(f"[{self.label}] Disconnected, reconnecting in 5s...")
                time.sleep(5)
