"""
Coconut Detection — Web Dashboard
Run: python app.py
Open: http://localhost:5001
"""

import os
import time
from flask import Flask, Response, render_template, jsonify, send_from_directory

from config import CAMERA_SOURCES, DETECTIONS_DIR, WEB_HOST, WEB_PORT
from camera_stream import CameraStream
from coconut_processor import CoconutProcessor

app = Flask(__name__)

# ─── Global state ────────────────────────────────────────────────────────────
cameras = {}
processor = None


def init_system():
    global cameras, processor

    print("=" * 60)
    print("  Coconut Detection — Web Dashboard")
    print("=" * 60)

    for src in CAMERA_SOURCES:
        cam = CameraStream(src["id"], src["url"], src["label"])
        cam.start()
        cameras[src["id"]] = cam
        print(f"  Starting {src['label']} ({src['id']})")

    processor = CoconutProcessor(cameras)
    processor.start()
    print(f"\n  Dashboard: http://localhost:{WEB_PORT}")
    print(f"  LAN access: http://<your-pc-ip>:{WEB_PORT}")
    print("=" * 60)


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("coconut_dashboard.html", cameras=CAMERA_SOURCES)


def generate_mjpeg(cam_id):
    while True:
        frame_bytes = processor.get_annotated_frame(cam_id)
        if frame_bytes:
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
        time.sleep(0.05)


@app.route("/video_feed/<cam_id>")
def video_feed(cam_id):
    if cam_id not in cameras:
        return "Camera not found", 404
    return Response(generate_mjpeg(cam_id),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/stats")
def api_stats():
    return jsonify(processor.get_stats())


@app.route("/api/detections")
def api_detections():
    return jsonify(processor.get_detection_history())


@app.route("/detections/<path:filename>")
def detection_image(filename):
    return send_from_directory(DETECTIONS_DIR, filename)


# ─── Entry ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    init_system()
    app.run(host=WEB_HOST, port=WEB_PORT, debug=False, threaded=True)
