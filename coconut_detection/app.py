"""
Coconut Maturity Classification System - Flask Web Application
Serves the dashboard, MJPEG video streams, and REST APIs for
maturity stats, distributions, trend data, and daily reports.

Inference powered by Roboflow "Coconut Maturity Detection" model.
Classes: Premature, Mature, Potential
"""

import os
import time
from flask import Flask, render_template, Response, jsonify, send_from_directory

from config import (
    WEB_HOST, WEB_PORT, CAMERA_SOURCES, DETECTIONS_DIR, REPORTS_DIR,
    INFERENCE_MODE,
)
from camera_stream import CameraStream
from coconut_processor import CoconutProcessor

app = Flask(__name__)

cameras = {}
processor = None


def init_system():
    """Initialize camera streams and the maturity processor."""
    global cameras, processor

    print(f"[System] Inference mode: {INFERENCE_MODE}")

    for src in CAMERA_SOURCES:
        cam = CameraStream(src["id"], src["url"], src["label"])
        cam.start()
        cameras[src["id"]] = cam

    processor = CoconutProcessor(cameras)
    processor.start()
    print(f"[System] Initialized {len(cameras)} cameras and processor")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def dashboard():
    """Serve the main dashboard page."""
    return render_template(
        "coconut_dashboard.html",
        cameras=CAMERA_SOURCES,
        inference_mode=INFERENCE_MODE,
    )


@app.route("/video_feed/<cam_id>")
def video_feed(cam_id):
    """MJPEG stream for a camera."""
    def generate():
        while True:
            frame_bytes = processor.get_annotated_frame(cam_id)
            if frame_bytes:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                )
            time.sleep(0.05)

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/stats")
def api_stats():
    """Current per-camera detection stats."""
    stats = processor.get_stats()
    total_premature = sum(s.get("premature", 0) for s in stats.values())
    total_mature = sum(s.get("mature", 0) for s in stats.values())
    total_potential = sum(s.get("potential", 0) for s in stats.values())
    cameras_online = sum(1 for s in stats.values() if s.get("connected"))

    return jsonify({
        "cameras": stats,
        "totals": {
            "premature": total_premature,
            "mature": total_mature,
            "potential": total_potential,
            "total": total_premature + total_mature + total_potential,
            "cameras_online": cameras_online,
            "cameras_total": len(cameras),
        },
    })


@app.route("/api/detections")
def api_detections():
    """Recent detection events."""
    return jsonify(processor.get_recent_detections(30))


@app.route("/api/distributions")
def api_distributions():
    """Per-camera maturity distribution (session totals)."""
    return jsonify(processor.get_tracker().get_all_distributions())


@app.route("/api/distributions/today")
def api_distributions_today():
    """Per-camera maturity distribution for today."""
    return jsonify(processor.get_tracker().get_today_distributions())


@app.route("/api/trends")
def api_trends():
    """Historical trend data for the last 14 days."""
    days = int(os.environ.get("TREND_DAYS", 14))
    return jsonify(processor.get_tracker().get_trend_data(days))


@app.route("/api/report/generate")
def api_generate_report():
    """Manually trigger a daily report."""
    report = processor.get_tracker().generate_daily_report()
    if report:
        return jsonify({"status": "ok", "report": report})
    return jsonify({"status": "no_data", "report": None})


@app.route("/detections/<filename>")
def serve_detection(filename):
    """Serve saved detection screenshot images."""
    return send_from_directory(DETECTIONS_DIR, filename)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    init_system()
    print(f"[Server] Starting on http://{WEB_HOST}:{WEB_PORT}")
    print(f"[Server] Dashboard: http://localhost:{WEB_PORT}")
    app.run(host=WEB_HOST, port=WEB_PORT, debug=False, threaded=True)
