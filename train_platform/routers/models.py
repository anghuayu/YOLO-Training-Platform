"""
Model management, evaluation, and export API routes.
"""
import json
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ..config import MODELS_DIR, EXPORTS_DIR, RUNS_DIR, DATASETS_DIR, EXPORT_FORMATS

router = APIRouter(prefix="/api/models", tags=["models"])


# --- Schemas ---
class ModelInfo(BaseModel):
    id: str
    name: str
    model_type: str
    path: str
    size_mb: float
    metrics: dict
    created_at: str


class EvaluateRequest(BaseModel):
    model_id: str
    dataset_id: str
    imgsz: int = 640
    conf: float = 0.25
    iou: float = 0.5
    device: str = "0"


class ExportRequest(BaseModel):
    model_id: str
    format: str = "onnx"
    imgsz: int = 640
    half: bool = False
    simplify: bool = True


# --- Helpers ---
def _load_model_meta(model_path: Path) -> dict:
    """Load model metadata from sidecar json."""
    meta_path = model_path.with_suffix(".json")
    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    return {}


def _save_model_meta(model_path: Path, meta: dict):
    """Save model metadata as sidecar json."""
    meta_path = model_path.with_suffix(".json")
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _scan_models() -> list:
    """Scan models directory and return model info list."""
    models = []
    if not MODELS_DIR.exists():
        return models
    for f in sorted(MODELS_DIR.iterdir(), reverse=True):
        if f.suffix in (".pt", ".pth", ".onnx", ".engine"):
            meta = _load_model_meta(f)
            size_mb = round(f.stat().st_size / (1024 * 1024), 2)

            # If sidecar JSON is empty, try to find run_meta.json as fallback
            if not meta:
                model_stem = f.stem
                # Look for matching run directory
                if RUNS_DIR.exists():
                    for run_dir in RUNS_DIR.iterdir():
                        if run_dir.is_dir() and not run_dir.name.startswith("smart_"):
                            run_meta_file = run_dir / "run_meta.json"
                            if run_meta_file.exists():
                                run_meta = json.loads(
                                    run_meta_file.read_text(encoding="utf-8")
                                )
                                # Match by model name or check if best_model points to this file
                                if (
                                    run_meta.get("model_name") == model_stem
                                    or str(f) in run_meta.get("best_model", "")
                                ):
                                    meta = {
                                        "name": run_meta.get("model_name", model_stem),
                                        "model_type": run_meta.get("model_type", "yolov8"),
                                        "dataset_name": run_meta.get("dataset_name", ""),
                                        "dataset_id": run_meta.get("dataset_id", ""),
                                        "metrics": run_meta.get("metrics", {}),
                                        "created_at": run_meta.get("created_at", ""),
                                        "description": "",
                                    }
                                    # Save sidecar for future use
                                    _save_model_meta(f, meta)
                                    break

            models.append(
                {
                    "id": f.stem,
                    "name": meta.get("name", f.stem),
                    "model_type": meta.get("model_type", "yolov8"),
                    "filename": f.name,
                    "path": str(f),
                    "size_mb": size_mb,
                    "metrics": meta.get("metrics", {}),
                    "dataset_name": meta.get("dataset_name", ""),
                    "dataset_id": meta.get("dataset_id", ""),
                    "created_at": meta.get("created_at", ""),
                    "description": meta.get("description", ""),
                }
            )
    return models


# --- Routes ---
@router.get("")
def list_models():
    """List all trained models."""
    return _scan_models()


@router.get("/{model_id}")
def get_model(model_id: str):
    """Get model details."""
    models = _scan_models()
    for m in models:
        if m["id"] == model_id:
            return m
    raise HTTPException(404, "Model not found")


@router.delete("/{model_id}")
def delete_model(model_id: str):
    """Delete a model."""
    # Find the model file (only match actual model files, not .json sidecar)
    for f in MODELS_DIR.iterdir():
        if f.stem == model_id and f.suffix in (".pt", ".pth", ".onnx", ".engine"):
            f.unlink()
            # Also delete sidecar json
            meta_path = f.with_suffix(".json")
            if meta_path.exists():
                meta_path.unlink()
            return {"message": "Model deleted"}
    raise HTTPException(404, "Model not found")


@router.put("/{model_id}")
def update_model(model_id: str, body: dict):
    """Update model metadata (name, description)."""
    for f in MODELS_DIR.iterdir():
        if f.stem == model_id and f.suffix in (".pt", ".pth", ".onnx", ".engine"):
            meta = _load_model_meta(f)
            if "name" in body:
                meta["name"] = body["name"]
            if "description" in body:
                meta["description"] = body["description"]
            _save_model_meta(f, meta)
            return {"message": "Model updated"}
    raise HTTPException(404, "Model not found")


@router.post("/{model_id}/download")
def download_model(model_id: str):
    """Download model file."""
    for f in MODELS_DIR.iterdir():
        if f.stem == model_id and f.suffix in (".pt", ".pth", ".onnx", ".engine"):
            return FileResponse(
                f,
                media_type="application/octet-stream",
                headers={"Content-Disposition": f'attachment; filename="{f.name}"'},
            )
    raise HTTPException(404, "Model not found")


# --- Evaluation ---
@router.post("/evaluate")
def evaluate_model(req: EvaluateRequest):
    """Evaluate a model on a dataset."""
    # Find model (only match actual model files, not .json sidecar)
    model_path = None
    for f in MODELS_DIR.iterdir():
        if f.stem == req.model_id and f.suffix in (".pt", ".pth", ".onnx", ".engine"):
            model_path = f
            break
    if not model_path:
        raise HTTPException(404, "Model not found")

    # Validate dataset
    ds_dir = DATASETS_DIR / req.dataset_id
    if not ds_dir.exists():
        raise HTTPException(404, "Dataset not found")

    # Generate data.yaml
    meta_file = ds_dir / "meta.json"
    if not meta_file.exists():
        raise HTTPException(404, "Dataset metadata not found")
    ds_meta = json.loads(meta_file.read_text(encoding="utf-8"))
    classes = ds_meta.get("classes", [])
    yaml_path = ds_dir / "data.yaml"
    yaml_content = f"path: {ds_dir}\ntrain: images\nval: images\n"
    yaml_content += f"nc: {len(classes)}\n"
    yaml_content += f"names: {list(classes)}\n"
    yaml_path.write_text(yaml_content, encoding="utf-8")

    # Get model type
    model_meta = _load_model_meta(model_path)
    model_type = model_meta.get("model_type", "yolov8")

    try:
        from ..models.factory import create_trainer

        trainer = create_trainer(model_type, req.model_id)
        results = trainer.evaluate(
            model_path=str(model_path),
            data_yaml=str(yaml_path),
            imgsz=req.imgsz,
            device=req.device,
        )

        # Save evaluation results
        eval_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        eval_dir = RUNS_DIR / "evaluations" / f"{req.model_id}_{eval_id}"
        eval_dir.mkdir(parents=True, exist_ok=True)
        (eval_dir / "results.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        return {
            "eval_id": eval_id,
            "model_id": req.model_id,
            "dataset_id": req.dataset_id,
            "metrics": results.get("metrics", {}),
            "per_class": results.get("per_class", {}),
        }

    except Exception as e:
        raise HTTPException(500, f"Evaluation failed: {str(e)}")


@router.get("/evaluations/{model_id}")
def get_evaluations(model_id: str):
    """Get evaluation history for a model."""
    eval_base = RUNS_DIR / "evaluations"
    if not eval_base.exists():
        return []

    evaluations = []
    for eval_dir in sorted(eval_base.iterdir(), reverse=True):
        if eval_dir.is_dir() and eval_dir.name.startswith(f"{model_id}_"):
            results_file = eval_dir / "results.json"
            if results_file.exists():
                results = json.loads(results_file.read_text(encoding="utf-8"))
                evaluations.append(
                    {
                        "eval_id": eval_dir.name.split("_", 1)[1] if "_" in eval_dir.name else eval_dir.name,
                        "timestamp": eval_dir.name,
                        "results": results,
                    }
                )
    return evaluations


# --- Export ---
@router.post("/export")
def export_model(req: ExportRequest):
    """Export model to a different format."""
    if req.format not in EXPORT_FORMATS:
        raise HTTPException(400, f"Unsupported format: {req.format}. Supported: {EXPORT_FORMATS}")

    # Find model (only match actual model files, not .json sidecar)
    model_path = None
    for f in MODELS_DIR.iterdir():
        if f.stem == req.model_id and f.suffix in (".pt", ".pth", ".onnx", ".engine"):
            model_path = f
            break
    if not model_path:
        raise HTTPException(404, "Model not found")

    # Get model type
    model_meta = _load_model_meta(model_path)
    model_type = model_meta.get("model_type", "yolov8")

    try:
        from ..models.factory import create_trainer

        trainer = create_trainer(model_type, req.model_id)
        export_path = trainer.export(
            model_path=str(model_path),
            format=req.format,
            imgsz=req.imgsz,
            half=req.half,
            simplify=req.simplify,
        )

        # Move exported file to exports directory
        export_dir = EXPORTS_DIR / req.model_id
        export_dir.mkdir(parents=True, exist_ok=True)
        dest = export_dir / Path(export_path).name
        shutil.copy2(export_path, dest)

        return {
            "message": "Model exported successfully",
            "format": req.format,
            "path": str(dest),
            "filename": dest.name,
            "download_url": f"/api/models/exports/{req.model_id}/{dest.name}",
        }

    except Exception as e:
        raise HTTPException(500, f"Export failed: {str(e)}")


@router.get("/exports/{model_id}")
def list_exports(model_id: str):
    """List exported files for a model."""
    export_dir = EXPORTS_DIR / model_id
    if not export_dir.exists():
        return []

    exports = []
    for f in sorted(export_dir.iterdir()):
        if f.is_file():
            exports.append(
                {
                    "filename": f.name,
                    "format": f.suffix[1:],
                    "size_mb": round(f.stat().st_size / (1024 * 1024), 2),
                    "created_at": datetime.fromtimestamp(f.stat().st_ctime).isoformat(),
                    "download_url": f"/api/models/exports/{model_id}/{f.name}",
                }
            )
    return exports


@router.get("/exports/{model_id}/{filename}")
def download_export(model_id: str, filename: str):
    """Download an exported model file."""
    file_path = EXPORTS_DIR / model_id / filename
    if not file_path.exists():
        raise HTTPException(404, "Export file not found")
    return FileResponse(
        file_path,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# --- Model types ---
@router.get("/types/available")
def get_available_model_types():
    """Get available model types."""
    from ..models.factory import get_available_types
    types = get_available_types()
    return {
        "types": [
            {
                "id": t,
                "name": t.upper() if "yolo" in t.lower() else t,
                "description": f"{t} model trainer",
            }
            for t in types
        ]
    }
