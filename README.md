# CCTV Face Recognition System

A real-time multi-camera face detection and recognition system with a web-based dashboard. Automatically detects, enrolls, and recognizes faces from RTSP WiFi cameras.

## Features

- **Multi-camera support** — monitors 2 RTSP cameras simultaneously
- **Auto-enrollment** — new faces are saved automatically after consistent detection
- **Face recognition** — recognizes returning people with green bounding boxes
- **Adaptive learning** — builds multiple embeddings per person over time for better accuracy
- **Web dashboard** — dark-themed CCTV-style UI accessible from any device on your network
- **Person tracking panel** — live status showing who is active, recently seen, or away
- **Detection logging** — all events logged to CSV with timestamps
- **Telegram alerts** — optional notifications when new faces are detected

## Requirements

- Python 3.11+
- Windows PC (tested with NVIDIA GPU, runs on CPU too)
- RTSP-capable WiFi cameras

## Quick Start

### 1. Install dependencies

```bash
cd face_recognition_cam
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure cameras

Edit `config.py` and set your camera RTSP URLs:

```python
CAMERA_SOURCES = [
    {"id": "cam1", "label": "Camera 1", "url": "rtsp://user:pass@192.168.1.10:554/stream1"},
    {"id": "cam2", "label": "Camera 2", "url": "rtsp://user:pass@192.168.1.11:554/stream1"},
]
```

### 3. Run

```bash
python app.py
```

Or double-click `start_dashboard.bat`.

### 4. Open dashboard

Navigate to `http://localhost:5000` in your browser. Access from other devices on your LAN using your PC's IP (shown in the terminal output).

## How It Works

1. **Camera streams** run in separate threads, continuously pulling frames via RTSP
2. **Face processor** alternates between cameras, running detection on each frame
3. **YuNet** (OpenCV DNN) detects faces with strict filtering (confidence, size, landmark validation)
4. **SFace** generates 128-dimensional face embeddings for recognition
5. **Multi-embedding matching** compares live faces against up to 5 stored embeddings per person
6. **Auto-enrollment** tracks unknown faces over multiple frames before saving (prevents false enrollments)
7. **Adaptive learning** periodically stores new embeddings from different angles to improve accuracy over time

## Project Structure

```
face_recognition_cam/
├── app.py               # Flask web server — main entry point
├── camera_stream.py     # Threaded RTSP camera reader
├── face_engine.py       # Face detection + recognition engine (OpenCV DNN)
├── face_processor.py    # Face processing thread for both cameras
├── config.py            # Camera URLs, thresholds, settings
├── name_faces.py        # CLI utility to rename/manage persons
├── requirements.txt     # Python dependencies
├── start_dashboard.bat  # Windows launcher script
├── face_database.json   # Person database (auto-generated)
├── detections.csv       # Detection event log (auto-generated)
├── templates/
│   └── dashboard.html   # Web dashboard UI
├── models/              # ONNX neural network models (auto-downloaded)
│   ├── face_detection_yunet_2023mar.onnx
│   └── face_recognition_sface_2021dec.onnx
└── known_faces/         # Stored face images
```

## Dashboard

The web dashboard shows:

- **Left panel** — live camera feeds with face detection bounding boxes
  - Green box = known person (with name and confidence %)
  - Orange box = scanning (unknown face being tracked)
  - Yellow box = newly enrolled person
- **Right panel** — person tracking list with thumbnails, status, and last-seen info
- **Top bar** — system stats (connected cameras, total persons, active count)

## Configuration

Key settings in `config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `PROCESS_EVERY_N_FRAMES` | 3 | Analyze every Nth frame (higher = less CPU) |
| `RESIZE_SCALE` | 0.25 | Shrink frames for processing (lower = faster) |
| `MATCH_TOLERANCE` | 0.5 | Face match strictness |
| `AUTO_ENROLL_CONFIDENCE_FRAMES` | 5 | Frames before auto-enrolling a new face |
| `NEW_FACE_COOLDOWN_SECONDS` | 10 | Cooldown between new enrollments |

Detection filtering in `face_engine.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `DETECT_CONFIDENCE` | 0.85 | YuNet confidence threshold (rejects non-face objects) |
| `MIN_FACE_SIZE` | 50 | Minimum face size in pixels |
| `COSINE_THRESHOLD` | 0.30 | Recognition match threshold |
| `MAX_EMBEDDINGS_PER_PERSON` | 5 | Embeddings stored per person for multi-angle matching |

## Managing Faces

### Add a known face manually

Place a photo in `known_faces/` (e.g., `john_smith.jpg`). The system picks it up automatically on next restart.

### Rename a person

```bash
python name_faces.py --rename P0001 "John Smith"
```

### List all persons

```bash
python name_faces.py --list
```

### Delete a person

```bash
python name_faces.py --delete P0003
```

## Telegram Alerts (Optional)

Enable notifications when new faces are detected:

1. Create a bot via [@BotFather](https://t.me/BotFather) on Telegram
2. Get your chat ID via [@userinfobot](https://t.me/userinfobot)
3. Edit `config.py`:

```python
TELEGRAM_ENABLED = True
TELEGRAM_BOT_TOKEN = "your-bot-token"
TELEGRAM_CHAT_ID = "your-chat-id"
```

## Tech Stack

- **OpenCV 4.13** — camera handling, DNN face detection (YuNet), face recognition (SFace)
- **Flask** — web server and MJPEG streaming
- **NumPy** — array operations
- **No dlib dependency** — uses pure OpenCV DNN models for maximum compatibility
