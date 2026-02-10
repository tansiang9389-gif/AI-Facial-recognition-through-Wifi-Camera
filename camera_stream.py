"""
Threaded RTSP Camera Stream Reader
Each camera runs in its own thread to prevent blocking.
"""

import cv2
import threading
import time


class CameraStream:
    def __init__(self, cam_id, url, label="Camera"):
        self.cam_id = cam_id
        self.url = url
        self.label = label
        self.frame = None
        self.lock = threading.Lock()
        self.running = False
        self.connected = False
        self._thread = None

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
                print(f"[{self.cam_id}] Cannot connect to {self.label}. Retrying in 5s...")
                self.connected = False
                time.sleep(5)
                continue

            print(f"[{self.cam_id}] Connected to {self.label}")
            self.connected = True
            fail_count = 0

            while self.running:
                ret, frame = cap.read()
                if not ret:
                    fail_count += 1
                    if fail_count > 30:
                        print(f"[{self.cam_id}] Lost connection. Reconnecting...")
                        break
                    time.sleep(0.05)
                    continue

                fail_count = 0
                with self.lock:
                    self.frame = frame

            cap.release()
            self.connected = False
            if self.running:
                time.sleep(2)
