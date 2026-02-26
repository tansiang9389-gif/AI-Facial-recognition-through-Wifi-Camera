"""
Multi-class Coconut Maturity YOLOv8 Training Script

Dataset: Roboflow "Coconut Maturity Detection"
  https://universe.roboflow.com/coconut-maturity-detection/coconut-maturity-detection
  3 classes: Premature (6462), Mature (3649), Potential (4103)
  3053 source images -> 7327 augmented (version 7)

Usage:
    python train_model.py
    python train_model.py --epochs 150 --batch 16
    python train_model.py --dataset-path /path/to/existing/dataset
"""

import os
import sys
import shutil
import argparse

from config import (
    MODELS_DIR, ROBOFLOW_API_KEY, ROBOFLOW_WORKSPACE,
    ROBOFLOW_PROJECT, ROBOFLOW_VERSION,
)


def download_dataset(api_key, workspace, project, version):
    """Download the Coconut Maturity Detection dataset from Roboflow."""
    from roboflow import Roboflow

    print(f"[Train] Connecting to Roboflow...")
    print(f"[Train] Workspace: {workspace}")
    print(f"[Train] Project: {project}")
    print(f"[Train] Version: {version}")

    rf = Roboflow(api_key=api_key)
    proj = rf.workspace(workspace).project(project)
    dataset = proj.version(version).download("yolov8")

    print(f"[Train] Dataset downloaded to: {dataset.location}")
    print(f"[Train] Classes: Premature, Mature, Potential")
    return dataset.location


def train(data_yaml_path, epochs=100, imgsz=640, batch=16, patience=15):
    """Train YOLOv8 nano model on the coconut maturity dataset."""
    from ultralytics import YOLO

    print("[Train] Loading YOLOv8n base model...")
    model = YOLO("yolov8n.pt")

    print(f"[Train] Starting training for {epochs} epochs...")
    print(f"[Train] Image size: {imgsz}, Batch: {batch}, Patience: {patience}")

    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name="coconut_maturity_detector",
        patience=patience,
        save=True,
        plots=True,
        verbose=True,
    )

    # Copy best weights to models directory
    best_pt = os.path.join("runs", "detect", "coconut_maturity_detector", "weights", "best.pt")
    if os.path.exists(best_pt):
        dest = os.path.join(MODELS_DIR, "coconut_maturity_best.pt")
        shutil.copy2(best_pt, dest)
        print(f"[Train] Best model saved to: {dest}")
    else:
        print("[Train] WARNING: best.pt not found in expected location")
        # Search for it
        for root, dirs, files in os.walk("runs"):
            if "best.pt" in files:
                src = os.path.join(root, "best.pt")
                dest = os.path.join(MODELS_DIR, "coconut_maturity_best.pt")
                shutil.copy2(src, dest)
                print(f"[Train] Found and saved: {src} -> {dest}")
                break

    return results


def main():
    parser = argparse.ArgumentParser(description="Train Coconut Maturity YOLOv8 Model")
    parser.add_argument("--api-key", default=ROBOFLOW_API_KEY,
                        help="Roboflow API key")
    parser.add_argument("--dataset-path", default=None,
                        help="Path to existing dataset (skip download)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Training epochs (default: 100)")
    parser.add_argument("--batch", type=int, default=16,
                        help="Batch size (default: 16)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Image size (default: 640)")
    parser.add_argument("--patience", type=int, default=15,
                        help="Early stopping patience (default: 15)")
    args = parser.parse_args()

    os.makedirs(MODELS_DIR, exist_ok=True)

    # Get dataset
    if args.dataset_path:
        dataset_path = args.dataset_path
        print(f"[Train] Using existing dataset: {dataset_path}")
        data_yaml = os.path.join(dataset_path, "data.yaml")
    else:
        if args.api_key == "YOUR_API_KEY_HERE":
            print("=" * 60)
            print("ERROR: Roboflow API key required for dataset download.")
            print()
            print("Get your key from the config.py or run:")
            print("  python train_model.py --api-key YOUR_KEY")
            print()
            print("Or provide your own dataset:")
            print("  python train_model.py --dataset-path /path/to/dataset")
            print()
            print("Dataset: Coconut Maturity Detection")
            print("  https://universe.roboflow.com/coconut-maturity-detection/coconut-maturity-detection")
            print("  Classes: Premature, Mature, Potential")
            print("=" * 60)
            sys.exit(1)

        dataset_path = download_dataset(
            args.api_key, ROBOFLOW_WORKSPACE, ROBOFLOW_PROJECT, ROBOFLOW_VERSION
        )
        data_yaml = os.path.join(dataset_path, "data.yaml")

    if not os.path.exists(data_yaml):
        print(f"[Train] ERROR: data.yaml not found at {data_yaml}")
        sys.exit(1)

    print(f"[Train] Using data.yaml: {data_yaml}")

    # Train
    train(data_yaml, args.epochs, args.imgsz, args.batch, args.patience)
    print("[Train] Training complete!")


if __name__ == "__main__":
    main()
