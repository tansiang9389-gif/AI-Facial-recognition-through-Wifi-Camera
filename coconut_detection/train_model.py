"""
Train YOLOv8 on Coconut Dataset from Roboflow
----------------------------------------------
This script downloads a coconut detection dataset from Roboflow Universe
and trains a YOLOv8 model on it.

Dataset: https://universe.roboflow.com/fruit-mtdup/coconut-a5ecn
         908 coconut images with annotations

Usage:
    1. Get a free API key from https://app.roboflow.com/settings/api
    2. Set ROBOFLOW_API_KEY in config.py (or pass via command line)
    3. Run: python train_model.py

The trained model will be saved to models/coconut_best.pt
"""

import os
import sys
import shutil


def main():
    from config import (
        ROBOFLOW_API_KEY,
        ROBOFLOW_WORKSPACE,
        ROBOFLOW_PROJECT,
        ROBOFLOW_VERSION,
        MODELS_DIR,
    )

    # Check API key
    api_key = ROBOFLOW_API_KEY
    if len(sys.argv) > 1:
        api_key = sys.argv[1]

    if api_key == "YOUR_API_KEY_HERE":
        print("=" * 60)
        print("  Roboflow API Key Required")
        print("=" * 60)
        print()
        print("  To train the coconut detection model, you need a Roboflow API key.")
        print()
        print("  Steps:")
        print("  1. Create a free account at https://app.roboflow.com")
        print("  2. Go to Settings -> API Keys")
        print("  3. Copy your API key")
        print("  4. Either:")
        print(f"     a. Set ROBOFLOW_API_KEY in config.py")
        print(f"     b. Run: python train_model.py YOUR_API_KEY")
        print()
        print("  Dataset: https://universe.roboflow.com/fruit-mtdup/coconut-a5ecn")
        print("  (908 coconut images with bounding box annotations)")
        print()
        sys.exit(1)

    # Import dependencies
    try:
        from roboflow import Roboflow
    except ImportError:
        print("Installing roboflow package...")
        os.system(f"{sys.executable} -m pip install roboflow")
        from roboflow import Roboflow

    try:
        from ultralytics import YOLO
    except ImportError:
        print("Installing ultralytics package...")
        os.system(f"{sys.executable} -m pip install ultralytics")
        from ultralytics import YOLO

    # Download dataset from Roboflow
    print("=" * 60)
    print("  Downloading Coconut Dataset from Roboflow...")
    print("=" * 60)

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
    version = project.version(ROBOFLOW_VERSION)
    dataset = version.download("yolov8")

    print(f"\n  Dataset downloaded to: {dataset.location}")
    print(f"  Training images: {dataset.location}/train")
    print(f"  Validation images: {dataset.location}/valid")

    # Train YOLOv8
    print("\n" + "=" * 60)
    print("  Training YOLOv8 on Coconut Dataset...")
    print("  This may take 30-60 minutes depending on your hardware.")
    print("=" * 60)

    model = YOLO("yolov8n.pt")  # Start from YOLOv8 nano pretrained on COCO

    results = model.train(
        data=os.path.join(dataset.location, "data.yaml"),
        epochs=50,
        imgsz=640,
        batch=16,
        name="coconut_detector",
        patience=10,         # Early stopping
        save=True,
        plots=True,
    )

    # Copy best model to models directory
    os.makedirs(MODELS_DIR, exist_ok=True)
    best_model_src = os.path.join("runs", "detect", "coconut_detector", "weights", "best.pt")

    if os.path.exists(best_model_src):
        dest = os.path.join(MODELS_DIR, "coconut_best.pt")
        shutil.copy2(best_model_src, dest)
        print(f"\n  Best model saved to: {dest}")
        print("  Training complete! You can now run: python app.py")
    else:
        print(f"\n  Warning: Could not find best.pt at {best_model_src}")
        print("  Check the runs/detect/coconut_detector/weights/ directory.")

    print("\n" + "=" * 60)
    print("  Training Results")
    print("=" * 60)
    print(f"  Results saved to: runs/detect/coconut_detector/")


if __name__ == "__main__":
    main()
