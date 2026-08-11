"""
Base model trainer interface.
All model types (YOLO, RT-DETR, etc.) must implement this interface.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from pathlib import Path


class BaseTrainer(ABC):
    """Abstract base class for all model trainers."""

    @abstractmethod
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name

    @abstractmethod
    def train(
        self,
        data_yaml: str,
        epochs: int = 100,
        imgsz: int = 640,
        batch: int = 16,
        output_dir: str = "",
        pretrained: str = "",
        progress_callback=None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Run training. Returns dict with training results/metrics.
        progress_callback(current_epoch, total_epochs, metrics) is called each epoch.
        """
        pass

    @abstractmethod
    def stop(self):
        """Stop current training gracefully."""
        pass

    @abstractmethod
    def evaluate(
        self, model_path: str, data_yaml: str, imgsz: int = 640, **kwargs
    ) -> Dict[str, Any]:
        """Evaluate a trained model. Returns metrics dict."""
        pass

    @abstractmethod
    def export(
        self, model_path: str, format: str = "onnx", imgsz: int = 640, **kwargs
    ) -> str:
        """Export model to specified format. Returns path to exported file."""
        pass

    @abstractmethod
    def predict(
        self,
        model_path: str,
        image_path: str,
        conf: float = 0.25,
        imgsz: int = 640,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Run inference on a single image. Returns list of detections."""
        pass

    @abstractmethod
    def get_available_models(self) -> List[str]:
        """Return list of available pretrained model sizes (e.g., ['n','s','m','l','x'])."""
        pass
