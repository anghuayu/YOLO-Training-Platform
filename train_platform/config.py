"""
YOLO Training Platform - Configuration
"""
import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent

# Data directories
DATA_DIR = BASE_DIR / "data"
DATASETS_DIR = DATA_DIR / "datasets"
MODELS_DIR = DATA_DIR / "models"
RUNS_DIR = DATA_DIR / "runs"
EXPORTS_DIR = DATA_DIR / "exports"

# Ensure directories exist
for d in [DATASETS_DIR, MODELS_DIR, RUNS_DIR, EXPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Server settings
HOST = "0.0.0.0"
PORT = 8080

# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

# Supported video extensions (for video-to-frames import)
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}

# Video import defaults
VIDEO_IMPORT_DEFAULTS = {
    "interval_sec": 1,       # Extract 1 frame every N seconds
    "jpeg_quality": 95,      # JPEG save quality (1-100)
    "max_frames": 500,       # Safety cap per video
}

# YOLO default classes (can be overridden per dataset)
DEFAULT_CLASSES = ["object"]

# Smart annotation defaults
SMART_ANNOTATION = {
    "min_images": 10,
    "default_epochs": 50,
    "default_imgsz": 640,
    "confidence_threshold": 0.25,
}

# Training defaults
TRAINING_DEFAULTS = {
    "epochs": 100,
    "imgsz": 640,
    "batch": 16,
    "lr0": 0.01,
    "patience": 50,
    "optimizer": "auto",
}

# Supported export formats
EXPORT_FORMATS = ["onnx", "torchscript", "tensorflow", "openvino", "engine"]

# Model type registry - add new model types here
MODEL_REGISTRY = {
    "yolov8": "train_platform.models.yolo_trainer.YOLOv8Trainer",
    # "yolov11": "train_platform.models.yolo_trainer.YOLOv11Trainer",
    # "rt-detr": "train_platform.models.rt_detr_trainer.RTDETRTrainer",
}
