# Coconut Detection via WiFi Camera

Real-time coconut detection system using **YOLOv8** and WiFi/RTSP cameras.
Built alongside the Facial Recognition system — runs independently on port 5001.

## Quick Start

### 1. Install Dependencies
```bash
cd coconut_detection
pip install -r requirements.txt
```

### 2. Get a Trained Model

**Option A: Train on Roboflow Coconut Dataset (Recommended)**
```bash
# Get a free API key from https://app.roboflow.com/settings/api
python train_model.py YOUR_ROBOFLOW_API_KEY
```
This downloads 908 coconut images from [Roboflow Universe](https://universe.roboflow.com/fruit-mtdup/coconut-a5ecn) and trains a YOLOv8 model.

**Option B: Quick Test with Base Model**
```bash
python download_model.py
```
Downloads YOLOv8n pre-trained on COCO (general object detection, not coconut-specific).

### 3. Configure Cameras
Edit `config.py` to set your RTSP camera URLs:
```python
CAMERA_SOURCES = [
    {"id": "cam1", "label": "Camera 1", "url": "rtsp://user:pass@192.168.1.10:554/stream1"},
]
```

### 4. Run
```bash
python app.py
```
Open http://localhost:5001 in your browser.

**Windows users:** Double-click `start_coconut_detection.bat`

## Dataset Source

The recommended training dataset is from Roboflow Universe:
- **Dataset:** [Coconut Object Detection](https://universe.roboflow.com/fruit-mtdup/coconut-a5ecn)
- **Images:** 908 annotated coconut images
- **Format:** YOLOv8 compatible

## Architecture

```
coconut_detection/
├── app.py                  # Flask web dashboard (port 5001)
├── camera_stream.py        # RTSP camera reader threads
├── coconut_detector.py     # YOLOv8 detection engine
├── coconut_processor.py    # Frame processing + annotation
├── config.py               # All settings
├── train_model.py          # Dataset download + training script
├── download_model.py       # Quick model download
├── requirements.txt        # Python dependencies
├── start_coconut_detection.bat  # Windows launcher
├── templates/
│   └── coconut_dashboard.html   # Web UI
├── models/
│   └── coconut_best.pt    # Trained model (after training)
└── detections/             # Saved detection screenshots
```

## Running Both Systems

You can run facial recognition and coconut detection simultaneously:

| System              | Port | Command                        |
|---------------------|------|--------------------------------|
| Facial Recognition  | 5000 | `python app.py` (root folder)  |
| Coconut Detection   | 5001 | `python app.py` (this folder)  |

Both share the same camera feeds but process them independently.

## Configuration

Key settings in `config.py`:

| Setting                  | Default | Description                          |
|--------------------------|---------|--------------------------------------|
| `CONFIDENCE_THRESHOLD`   | 0.40    | Min confidence for detection         |
| `IOU_THRESHOLD`          | 0.45    | NMS IoU threshold                    |
| `PROCESS_EVERY_N_FRAMES` | 2       | Skip frames for performance          |
| `SAVE_DETECTIONS`        | True    | Save detection screenshots           |
| `SAVE_COOLDOWN_SECONDS`  | 5       | Min time between screenshots         |
| `TELEGRAM_ENABLED`       | False   | Enable Telegram alerts               |
