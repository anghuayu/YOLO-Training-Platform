# YOLO Training Platform

A complete web-based platform for YOLO model training, annotation, and evaluation.

## Features

- **Dataset Management**: Upload, organize, and manage image datasets with YOLO-format annotations
- **Video Import**: Import video files (MP4/AVI/MOV/MKV/etc.), automatically extract frames at configurable intervals and save as JPG
- **Version Management**: Create dataset snapshots (images + labels), restore or delete versions for safe iteration
- **Annotation Tool**: Canvas-based bounding box annotation with keyboard shortcuts, zoom/pan, and class management
- **Dynamic Class Editing**: Add/edit class labels from both the dataset page and the annotation page, persisted to dataset metadata
- **Smart Annotation**: Auto-annotate remaining images by training a temporary model on manually labeled data
- **Model Training**: Configure and launch YOLOv8 training with hyperparameter tuning, real-time progress monitoring, and loss curves
- **Model Management**: View, export (ONNX/TensorFlow/TensorRT/etc.), download, and organize trained models
- **Model Evaluation**: Evaluate models on datasets with mAP, precision, recall metrics and per-class analysis
- **Extensible Architecture**: Factory pattern makes it easy to add new model types (RT-DETR, YOLOv11, etc.)

## Quick Start

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Run

**Windows:**
```bash
start.bat
```

**Linux/Mac:**
```bash
chmod +x start.sh
./start.sh
```

**Manual:**
```bash
python main.py
```

After starting, the platform will display the access URL:
```
============================================================
  YOLO Training Platform
============================================================
  Local:   http://localhost:8080
  Network: http://192.168.x.x:8080
============================================================
```

## Project Structure

```
yolo_training_platform/
├── main.py                     # FastAPI entry point
├── requirements.txt            # Python dependencies
├── start.bat / start.sh        # Startup scripts
├── train_platform/             # Backend package
│   ├── config.py               # Configuration (image/video extensions, defaults)
│   ├── models/                 # Model abstraction layer
│   │   ├── base_trainer.py     # Abstract base class (interface)
│   │   ├── yolo_trainer.py     # YOLOv8 implementation
│   │   └── factory.py          # Factory pattern for model creation
│   └── routers/                # API route handlers
│       ├── datasets.py         # Dataset CRUD + image upload + video import + versions
│       ├── annotations.py      # Annotation CRUD + smart annotation
│       ├── training.py         # Training job management
│       └── models.py           # Model management + evaluation + export
├── static/                     # Frontend
│   ├── index.html              # Dashboard
│   ├── datasets.html           # Dataset management (+ video import, versions, edit)
│   ├── annotate.html           # Annotation tool (+ inline class adding)
│   ├── train.html              # Training interface
│   ├── models.html             # Model management
│   ├── evaluate.html           # Model evaluation
│   ├── css/style.css           # Global styles
│   └── js/app.js               # Shared JavaScript
└── data/                       # Data storage (auto-created)
    ├── datasets/               # Uploaded datasets
    │   └── {dataset_id}/
    │       ├── meta.json       # Metadata (name, classes, versions, video_imports)
    │       ├── images/         # Image files (JPG/PNG/etc.)
    │       ├── labels/         # YOLO-format annotation .txt files
    │       ├── videos/         # Original imported video files
    │       └── versions/       # Version snapshots
    │           └── v{N}_{timestamp}/
    │               ├── images/
    │               └── labels/
    ├── models/                 # Trained models
    ├── runs/                   # Training runs
    └── exports/                # Exported models
```

## API Reference

### Datasets
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/datasets` | List all datasets |
| POST | `/api/datasets` | Create dataset |
| GET | `/api/datasets/{id}` | Get dataset details |
| PUT | `/api/datasets/{id}` | Update dataset (name/classes/description) |
| DELETE | `/api/datasets/{id}` | Delete dataset |
| POST | `/api/datasets/{id}/images` | Upload images |
| GET | `/api/datasets/{id}/images` | List images |
| GET | `/api/datasets/{id}/images/{name}` | Get image file |
| DELETE | `/api/datasets/{id}/images/{name}` | Delete image |
| GET | `/api/datasets/{id}/export-yolo` | Export as YOLO zip |
| POST | `/api/datasets/{id}/import-video` | Upload video & extract frames as JPG |
| GET | `/api/datasets/{id}/videos` | List imported videos |
| GET | `/api/datasets/{id}/videos/{name}` | Get video file |
| POST | `/api/datasets/{id}/versions` | Create version snapshot |
| GET | `/api/datasets/{id}/versions` | List all versions |
| POST | `/api/datasets/{id}/versions/{vid}/restore` | Restore a version |
| DELETE | `/api/datasets/{id}/versions/{vid}` | Delete a version |

### Annotations
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/annotations/{dataset_id}` | List all annotations |
| GET | `/api/annotations/{dataset_id}/{image}` | Get image annotations |
| PUT | `/api/annotations/{dataset_id}/{image}` | Save annotations |
| DELETE | `/api/annotations/{dataset_id}/{image}` | Delete annotations |
| POST | `/api/annotations/{dataset_id}/smart-annotate` | Start smart annotation |
| GET | `/api/annotations/{dataset_id}/smart-annotate/status` | Get smart annotation status |

### Training
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/training` | List all training runs |
| GET | `/api/training/active` | Get active training |
| POST | `/api/training` | Start training |
| GET | `/api/training/{run_id}` | Get training status |
| POST | `/api/training/{run_id}/stop` | Stop training |
| DELETE | `/api/training/{run_id}` | Delete training run |
| GET | `/api/training/{run_id}/logs` | Get training logs |

### Models
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/models` | List all models |
| GET | `/api/models/{id}` | Get model details |
| PUT | `/api/models/{id}` | Update model metadata |
| DELETE | `/api/models/{id}` | Delete model |
| POST | `/api/models/{id}/download` | Download model file |
| POST | `/api/models/evaluate` | Evaluate model |
| GET | `/api/models/evaluations/{id}` | Get evaluation history |
| POST | `/api/models/export` | Export model |
| GET | `/api/models/exports/{id}` | List exports |
| GET | `/api/models/types/available` | Get available model types |

### Dashboard
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/dashboard` | Get dashboard statistics |

## Adding New Model Types

1. Create a new trainer class in `train_platform/models/` implementing `BaseTrainer`
2. Register it in `train_platform/config.py` under `MODEL_REGISTRY`
3. The frontend will automatically pick up the new model type

Example:
```python
# train_platform/models/rt_detr_trainer.py
from .base_trainer import BaseTrainer

class RTDETRTrainer(BaseTrainer):
    def train(self, ...): ...
    def evaluate(self, ...): ...
    def export(self, ...): ...
    def predict(self, ...): ...
    def stop(self): ...
    def get_available_models(self): ...
```

```python
# In config.py MODEL_REGISTRY:
"rt-detr": "train_platform.models.rt_detr_trainer.RTDETRTrainer",
```

## Keyboard Shortcuts (Annotation Tool)

| Key | Action |
|-----|--------|
| D | Draw box tool |
| S | Select tool |
| P | Pan tool |
| A | Previous image |
| D | Next image |
| Delete | Delete selected box |
| Escape | Deselect |
| Scroll | Zoom in/out |

## Server Deployment

For production deployment, use a proper ASGI server:

```bash
# Using uvicorn with multiple workers
uvicorn main:app --host 0.0.0.0 --port 8080 --workers 4

# Using gunicorn (Linux)
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8080
```

Consider adding:
- Nginx reverse proxy
- SSL/TLS certificates
- Authentication middleware
- File size limits
- GPU resource management
