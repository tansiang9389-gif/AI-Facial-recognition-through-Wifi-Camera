"""
CCTV Face Recognition — Web Dashboard
Run: python app.py
Open: http://localhost:5000
"""

import os
import time
from flask import Flask, Response, render_template, jsonify, send_from_directory

from config import CAMERA_SOURCES, KNOWN_FACES_DIR, WEB_HOST, WEB_PORT
from camera_stream import CameraStream
from face_processor import FaceProcessor

app = Flask(__name__)

# ─── Global state ────────────────────────────────────────────────────────────
cameras = {}
processor = None


def init_system():
    global cameras, processor

    print("=" * 60)
    print("  CCTV Face Recognition — Web Dashboard")
    print("=" * 60)

    for src in CAMERA_SOURCES:
        cam = CameraStream(src["id"], src["url"], src["label"])
        cam.start()
        cameras[src["id"]] = cam
        print(f"  Starting {src['label']} ({src['id']})")

    processor = FaceProcessor(cameras)
    processor.start()
    print(f"\n  Dashboard: http://localhost:{WEB_PORT}")
    print(f"  LAN access: http://<your-pc-ip>:{WEB_PORT}")
    print("=" * 60)


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("dashboard.html", cameras=CAMERA_SOURCES)


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


@app.route("/api/persons")
def api_persons():
    # Combine database info with live status
    db_persons = processor.get_db_persons()
    live_status = processor.get_persons()
    now = time.time()

    result = []
    for pid, info in sorted(db_persons.items()):
        status = live_status.get(pid, {})
        last_ts = status.get("last_seen_ts", 0)

        if last_ts == 0:
            # Never seen live this session
            state = "away"
        elif now - last_ts < 10:
            state = "active"
        elif now - last_ts < 60:
            state = "recent"
        else:
            state = "away"

        result.append({
            "pid": pid,
            "name": info["name"],
            "image": info["image"],
            "first_seen": info["first_seen"],
            "last_seen": status.get("last_seen", ""),
            "camera": status.get("camera", ""),
            "camera_id": status.get("camera_id", ""),
            "confidence": status.get("confidence", 0),
            "state": state,
        })

    return jsonify(result)


@app.route("/api/stats")
def api_stats():
    return jsonify(processor.get_stats())


@app.route("/known_faces/<path:filename>")
def known_face_image(filename):
    return send_from_directory(KNOWN_FACES_DIR, filename)


# ─── Entry ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    init_system()
    app.run(host=WEB_HOST, port=WEB_PORT, debug=False, threaded=True)
