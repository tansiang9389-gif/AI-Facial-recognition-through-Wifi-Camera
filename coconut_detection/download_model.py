"""
Model Download Utility for Coconut Maturity Classification.

Downloads the trained YOLOv8 model from Roboflow or falls back
to the base YOLOv8n model.

Dataset: Coconut Maturity Detection (Premature, Mature, Potential)
https://universe.roboflow.com/coconut-maturity-detection/coconut-maturity-detection
"""

import os
import sys

from config import (
    MODELS_DIR, ROBOFLOW_API_KEY, ROBOFLOW_WORKSPACE,
    ROBOFLOW_PROJECT, ROBOFLOW_MODEL_VERSION,
)


def download_from_roboflow(api_key):
    """Attempt to download the trained model weights from Roboflow."""
    try:
        from roboflow import Roboflow
        rf = Roboflow(api_key=api_key)
        project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
        version = project.version(ROBOFLOW_MODEL_VERSION)

        # Try to download model weights
        print(f"[Download] Downloading model from Roboflow v{ROBOFLOW_MODEL_VERSION}...")
        model = version.model
        if model:
            print("[Download] Model available via Roboflow Hosted API")
            print("[Download] No local .pt download needed for hosted inference mode")
            return True
        return False
    except Exception as e:
        print(f"[Download] Roboflow error: {e}")
        return False


def download_base_model():
    """Download the base YOLOv8n model from Ultralytics."""
    from ultralytics import YOLO

    print("[Download] Downloading YOLOv8n base model...")
    model = YOLO("yolov8n.pt")

    dest = os.path.join(MODELS_DIR, "coconut_maturity_best.pt")
    model_path = "yolov8n.pt"
    if os.path.exists(model_path):
        import shutil
        shutil.copy2(model_path, dest)
        print(f"[Download] Base model saved to: {dest}")
        print()
        print("=" * 60)
        print("NOTE: This is the generic YOLOv8n model (COCO 80-class).")
        print("It will NOT detect coconut maturity levels accurately.")
        print()
        print("For accurate detection, use INFERENCE_MODE = 'roboflow_hosted'")
        print("in config.py (default). This uses the Roboflow cloud API with")
        print("a pre-trained model (mAP: 78.3%, classes: Premature/Mature/Potential)")
        print()
        print("To train a local model, run:")
        print("  python train_model.py")
        print("=" * 60)
        return True
    return False


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    dest = os.path.join(MODELS_DIR, "coconut_maturity_best.pt")
    if os.path.exists(dest):
        print(f"[Download] Model already exists: {dest}")
        return

    # Try Roboflow
    if ROBOFLOW_API_KEY and ROBOFLOW_API_KEY != "YOUR_API_KEY_HERE":
        if download_from_roboflow(ROBOFLOW_API_KEY):
            return

    # Fallback to base model
    if not download_base_model():
        print("[Download] ERROR: Could not download any model.")
        sys.exit(1)


if __name__ == "__main__":
    main()
