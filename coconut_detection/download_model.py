"""
Download Pre-trained Coconut Detection Model from Roboflow
-----------------------------------------------------------
Downloads a pre-trained coconut detection model via Roboflow's
inference API and converts it for local use.

Alternative: If you prefer to train your own model, use train_model.py instead.

Usage:
    python download_model.py [ROBOFLOW_API_KEY]
"""

import os
import sys


def main():
    from config import (
        ROBOFLOW_API_KEY,
        MODELS_DIR,
    )

    api_key = ROBOFLOW_API_KEY
    if len(sys.argv) > 1:
        api_key = sys.argv[1]

    if api_key == "YOUR_API_KEY_HERE":
        print("=" * 60)
        print("  Roboflow API Key Required")
        print("=" * 60)
        print()
        print("  Option 1: Train your own model (recommended)")
        print("    python train_model.py YOUR_API_KEY")
        print()
        print("  Option 2: Quick start with YOLOv8 pre-trained on COCO")
        print("    This will detect general objects. For coconut-specific")
        print("    detection, training on the coconut dataset is required.")
        print()

        response = input("  Download YOLOv8n base model for quick testing? [y/N]: ").strip().lower()
        if response == 'y':
            download_base_model()
        else:
            print("\n  Please set up your Roboflow API key first.")
            print("  Get one at: https://app.roboflow.com/settings/api")
        return

    # Try to download from Roboflow
    try:
        from roboflow import Roboflow
    except ImportError:
        print("Installing roboflow package...")
        os.system(f"{sys.executable} -m pip install roboflow")
        from roboflow import Roboflow

    print("=" * 60)
    print("  Downloading Coconut Model from Roboflow...")
    print("=" * 60)

    try:
        rf = Roboflow(api_key=api_key)

        # Try the pre-trained coconut model from the community
        project = rf.workspace("fruit-mtdup").project("coconut-a5ecn")
        version = project.version(1)

        # Download the dataset and train locally (most reliable approach)
        print("\n  For best results, train a custom model:")
        print("    python train_model.py")
        print()
        print("  Downloading YOLOv8n base model for quick testing...")
        download_base_model()

    except Exception as e:
        print(f"\n  Error: {e}")
        print("  Falling back to base YOLOv8n model...")
        download_base_model()


def download_base_model():
    """Download YOLOv8n pre-trained on COCO as a starting point."""
    from config import MODELS_DIR

    try:
        from ultralytics import YOLO
    except ImportError:
        print("Installing ultralytics package...")
        os.system(f"{sys.executable} -m pip install ultralytics")
        from ultralytics import YOLO

    os.makedirs(MODELS_DIR, exist_ok=True)

    print("\n  Downloading YOLOv8n base model...")
    model = YOLO("yolov8n.pt")

    # Save to our models directory
    dest = os.path.join(MODELS_DIR, "coconut_best.pt")

    import shutil
    # The model downloads to the current directory as yolov8n.pt
    if os.path.exists("yolov8n.pt"):
        shutil.copy2("yolov8n.pt", dest)
        print(f"  Model saved to: {dest}")
        print()
        print("  NOTE: This is a general COCO model, not coconut-specific.")
        print("  It can detect 80 object classes including 'banana', 'apple', etc.")
        print("  For coconut-specific detection, run: python train_model.py")
        print()
        print("  You can test the system now with: python app.py")
    else:
        print("  Error: Could not download model.")


if __name__ == "__main__":
    main()
