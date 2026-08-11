"""
Model factory - creates trainer instances based on model type.
To add a new model type:
  1. Create a new trainer class implementing BaseTrainer
  2. Register it in config.py MODEL_REGISTRY
"""
import importlib
from typing import Dict, Any

from ..config import MODEL_REGISTRY
from .base_trainer import BaseTrainer


def create_trainer(model_type: str, model_name: str, **kwargs) -> BaseTrainer:
    """Create a trainer instance for the given model type."""
    if model_type not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unknown model type: {model_type}. Available: {available}"
        )

    module_path, class_name = MODEL_REGISTRY[model_type].rsplit(".", 1)
    module = importlib.import_module(module_path)
    trainer_class = getattr(module, class_name)
    return trainer_class(model_name=model_name, **kwargs)


def get_available_types() -> list:
    """Return list of registered model types."""
    return list(MODEL_REGISTRY.keys())
