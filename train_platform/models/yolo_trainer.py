"""
YOLOv8 trainer implementation using Ultralytics.
To add a new model type, create a new file similar to this one
and register it in config.py MODEL_REGISTRY.
"""
import shutil
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List

from ultralytics import YOLO

from .base_trainer import BaseTrainer


class YOLOv8Trainer(BaseTrainer):
    """YOLOv8 trainer using Ultralytics library."""

    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.model: Optional[YOLO] = None
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------
    def train(
        self,
        data_yaml: str,
        epochs: int = 100,
        imgsz: int = 640,
        batch: int = 16,
        output_dir: str = "",
        pretrained: str = "",
        patience: int = 50,
        lr0: float = 0.01,
        optimizer: str = "auto",
        device: str = "0",
        progress_callback=None,
        **kwargs,
    ) -> Dict[str, Any]:
        # Load or create model
        if pretrained and Path(pretrained).exists():
            self.model = YOLO(pretrained)
        else:
            self.model = YOLO("yolov8n.pt")  # default small model

        # Register per-epoch progress callback
        if progress_callback:
            def _on_epoch_end(trainer):
                current_epoch = trainer.epoch + 1  # 0-indexed → 1-indexed
                total_epochs = trainer.epochs
                # Extract partial metrics from trainer
                epoch_metrics = {}
                if hasattr(trainer, "metrics") and trainer.metrics:
                    try:
                        m = trainer.metrics
                        epoch_metrics = {
                            "map50": float(m.get("metrics/mAP50(B)", 0)) if isinstance(m, dict) else 0,
                            "map50_95": float(m.get("metrics/mAP50-95(B)", 0)) if isinstance(m, dict) else 0,
                            "precision": float(m.get("metrics/precision(B)", 0)) if isinstance(m, dict) else 0,
                            "recall": float(m.get("metrics/recall(B)", 0)) if isinstance(m, dict) else 0,
                        }
                    except Exception:
                        pass
                progress_callback(current_epoch, total_epochs, epoch_metrics)

            self.model.add_callback("on_train_epoch_end", _on_epoch_end)

        train_args = {
            "data": data_yaml,
            "epochs": epochs,
            "imgsz": imgsz,
            "batch": batch,
            "project": output_dir,
            "name": self.model_name,
            "exist_ok": True,
            "patience": patience,
            "lr0": lr0,
            "optimizer": optimizer,
            "device": device,
            "verbose": True,
            "plots": True,
        }
        # Merge any extra kwargs
        for k, v in kwargs.items():
            if k not in train_args:
                train_args[k] = v

        results = self.model.train(**train_args)

        # Return key metrics
        metrics = {}
        if hasattr(self.model, "metrics"):
            m = self.model.metrics
            metrics = {
                "map50": float(m.box.map50) if hasattr(m.box, "map50") else 0,
                "map50_95": float(m.box.map) if hasattr(m.box, "map") else 0,
                "precision": float(m.box.mp) if hasattr(m.box, "mp") else 0,
                "recall": float(m.box.mr) if hasattr(m.box, "mr") else 0,
            }

        best_model = str(
            Path(output_dir) / self.model_name / "weights" / "best.pt"
        )
        return {
            "best_model": best_model,
            "metrics": metrics,
            "results_dir": str(Path(output_dir) / self.model_name),
        }

    # ------------------------------------------------------------------
    def stop(self):
        if self.model:
            self.model.trainer._stop = True

    # ------------------------------------------------------------------
    def evaluate(
        self, model_path: str, data_yaml: str, imgsz: int = 640, **kwargs
    ) -> Dict[str, Any]:
        model = YOLO(model_path)
        results = model.val(data=data_yaml, imgsz=imgsz, device=kwargs.get("device", "0"))

        metrics = {
            "map50": float(results.box.map50) if hasattr(results.box, "map50") else 0,
            "map50_95": float(results.box.map) if hasattr(results.box, "map") else 0,
            "map75": float(results.box.map75) if hasattr(results.box, "map75") else 0,
            "precision": float(results.box.mp) if hasattr(results.box, "mp") else 0,
            "recall": float(results.box.mr) if hasattr(results.box, "mr") else 0,
        }

        # Per-class metrics
        per_class = {}
        if hasattr(results.box, "ap50") and results.box.ap50 is not None:
            for i, ap in enumerate(results.box.ap50):
                per_class[f"class_{i}"] = {"ap50": float(ap)}

        return {"metrics": metrics, "per_class": per_class}

    # ------------------------------------------------------------------
    def export(
        self, model_path: str, format: str = "onnx", imgsz: int = 640, **kwargs
    ) -> str:
        model = YOLO(model_path)
        exported = model.export(format=format, imgsz=imgsz)
        return str(exported)

    # ------------------------------------------------------------------
    def predict(
        self,
        model_path: str,
        image_path: str,
        conf: float = 0.25,
        imgsz: int = 640,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        model = YOLO(model_path)
        results = model(image_path, conf=conf, imgsz=imgsz, verbose=False)
        detections = []
        if results and len(results) > 0:
            r = results[0]
            if r.boxes is not None:
                for box in r.boxes:
                    xyxy = box.xyxy[0].tolist()
                    detections.append(
                        {
                            "bbox": xyxy,  # [x1, y1, x2, y2]
                            "confidence": float(box.conf[0]),
                            "class_id": int(box.cls[0]),
                            "class_name": r.names[int(box.cls[0])],
                        }
                    )
        return detections

    # ------------------------------------------------------------------
    def get_available_models(self) -> List[str]:
        return ["n", "s", "m", "l", "x"]
