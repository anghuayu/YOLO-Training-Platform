"""
Training management API routes.
"""
import json
import uuid
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..config import MODELS_DIR, RUNS_DIR, DATASETS_DIR, TRAINING_DEFAULTS

router = APIRouter(prefix="/api/training", tags=["training"])


# --- Schemas ---
class TrainingRequest(BaseModel):
    dataset_id: str
    model_type: str = "yolov8"
    model_name: str = ""
    pretrained: str = ""
    epochs: int = 100
    imgsz: int = 640
    batch: int = 16
    lr0: float = 0.01
    patience: int = 50
    optimizer: str = "auto"
    device: str = "0"


class TrainingUpdate(BaseModel):
    status: Optional[str] = None


# --- State management ---
_training_tasks = {}


def _save_run_meta(run_dir: Path, meta: dict):
    (run_dir / "run_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _load_run_meta(run_dir: Path) -> dict:
    meta_file = run_dir / "run_meta.json"
    if meta_file.exists():
        return json.loads(meta_file.read_text(encoding="utf-8"))
    return {}


# --- Routes ---
@router.get("")
def list_training_runs():
    """List all training runs."""
    runs = []
    if not RUNS_DIR.exists():
        return runs
    for run_dir in sorted(RUNS_DIR.iterdir(), reverse=True):
        if run_dir.is_dir() and not run_dir.name.startswith("smart_"):
            meta = _load_run_meta(run_dir)
            if meta:
                runs.append(meta)
    return runs


@router.get("/active")
def get_active_training():
    """Get currently active training task."""
    for tid, task in _training_tasks.items():
        if task["status"] in ("training", "queued"):
            return task
    return {"status": "none"}


@router.post("")
def start_training(req: TrainingRequest):
    """Start a new training run."""
    # Check if there's already an active training
    for tid, task in _training_tasks.items():
        if task["status"] == "training":
            raise HTTPException(400, "A training run is already in progress")

    # Validate dataset
    ds_dir = DATASETS_DIR / req.dataset_id
    if not ds_dir.exists():
        raise HTTPException(404, "Dataset not found")

    meta_file = ds_dir / "meta.json"
    if not meta_file.exists():
        raise HTTPException(404, "Dataset metadata not found")

    ds_meta = json.loads(meta_file.read_text(encoding="utf-8"))

    # Check dataset has images
    from ..config import IMAGE_EXTENSIONS
    img_dir = ds_dir / "images"
    image_count = sum(1 for f in img_dir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS) if img_dir.exists() else 0
    if image_count == 0:
        raise HTTPException(400, "Dataset has no images")

    # Generate data.yaml
    classes = ds_meta.get("classes", [])
    yaml_path = ds_dir / "data.yaml"
    yaml_content = f"path: {ds_dir}\ntrain: images\nval: images\n"
    yaml_content += f"nc: {len(classes)}\n"
    yaml_content += f"names: {list(classes)}\n"
    yaml_path.write_text(yaml_content, encoding="utf-8")

    # Resolve pretrained model path
    pretrained_path = req.pretrained
    if pretrained_path and not Path(pretrained_path).exists():
        # It might be a model ID, look it up
        model_file = MODELS_DIR / f"{pretrained_path}.pt"
        if model_file.exists():
            pretrained_path = str(model_file)
        else:
            # Try to find any file matching the ID
            found = False
            if MODELS_DIR.exists():
                for f in MODELS_DIR.iterdir():
                    if pretrained_path in f.stem:
                        pretrained_path = str(f)
                        found = True
                        break
            if not found:
                pretrained_path = ""  # Invalid path, will use default

    # Create training run
    run_id = uuid.uuid4().hex[:12]
    model_name = req.model_name or f"{req.model_type}_{run_id}"
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True)

    run_meta = {
        "id": run_id,
        "model_type": req.model_type,
        "model_name": model_name,
        "dataset_id": req.dataset_id,
        "dataset_name": ds_meta.get("name", ""),
        "status": "queued",
        "progress": 0,
        "epoch": 0,
        "total_epochs": req.epochs,
        "params": {
            "epochs": req.epochs,
            "imgsz": req.imgsz,
            "batch": req.batch,
            "lr0": req.lr0,
            "patience": req.patience,
            "optimizer": req.optimizer,
            "device": req.device,
        },
        "created_at": datetime.now().isoformat(),
        "started_at": "",
        "completed_at": "",
        "best_model": "",
        "metrics": {},
        "error": "",
    }
    _save_run_meta(run_dir, run_meta)

    _training_tasks[run_id] = {
        "id": run_id,
        "status": "training",
        "progress": 0,
        "epoch": 0,
        "total_epochs": req.epochs,
        "model_name": model_name,
        "model_type": req.model_type,
        "message": "Starting training...",
    }

    def _run_training():
        try:
            from ..models.factory import create_trainer

            trainer = create_trainer(req.model_type, model_name)
            _training_tasks[run_id]["message"] = "Training in progress..."

            # Update meta
            run_meta["status"] = "training"
            run_meta["started_at"] = datetime.now().isoformat()
            _save_run_meta(run_dir, run_meta)

            # Progress callback — called by trainer after each epoch
            def _progress_cb(current_epoch, total_epochs, metrics):
                pct = round(current_epoch / total_epochs * 100, 1) if total_epochs else 0
                _training_tasks[run_id]["epoch"] = current_epoch
                _training_tasks[run_id]["total_epochs"] = total_epochs
                _training_tasks[run_id]["progress"] = pct
                _training_tasks[run_id]["message"] = f"Epoch {current_epoch}/{total_epochs}"
                if metrics:
                    _training_tasks[run_id]["metrics"] = metrics

            result = trainer.train(
                data_yaml=str(yaml_path),
                epochs=req.epochs,
                imgsz=req.imgsz,
                batch=req.batch,
                output_dir=str(run_dir),
                pretrained=pretrained_path,
                patience=req.patience,
                lr0=req.lr0,
                optimizer=req.optimizer,
                device=req.device,
                progress_callback=_progress_cb,
            )

            # Training completed successfully
            best_model = result.get("best_model", "")
            metrics = result.get("metrics", {})

            # Copy best model to models directory
            model_dest = MODELS_DIR / f"{model_name}.pt"
            if best_model and Path(best_model).exists():
                import shutil
                shutil.copy2(best_model, model_dest)

                # Write sidecar metadata for model management page
                model_meta = {
                    "name": model_name,
                    "model_type": req.model_type,
                    "dataset_name": ds_meta.get("name", ""),
                    "dataset_id": req.dataset_id,
                    "metrics": metrics,
                    "created_at": datetime.now().isoformat(),
                    "description": "",
                }
                meta_path = model_dest.with_suffix(".json")
                meta_path.write_text(
                    json.dumps(model_meta, ensure_ascii=False, indent=2), encoding="utf-8"
                )

            # Update run meta
            run_meta["status"] = "completed"
            run_meta["completed_at"] = datetime.now().isoformat()
            run_meta["progress"] = 100
            run_meta["best_model"] = str(model_dest)
            run_meta["metrics"] = metrics
            run_meta["results_dir"] = result.get("results_dir", "")
            _save_run_meta(run_dir, run_meta)

            _training_tasks[run_id]["status"] = "completed"
            _training_tasks[run_id]["progress"] = 100
            _training_tasks[run_id]["message"] = "Training completed!"
            _training_tasks[run_id]["metrics"] = metrics

        except Exception as e:
            run_meta["status"] = "error"
            run_meta["error"] = str(e)
            run_meta["completed_at"] = datetime.now().isoformat()
            _save_run_meta(run_dir, run_meta)

            _training_tasks[run_id]["status"] = "error"
            _training_tasks[run_id]["message"] = f"Training failed: {str(e)}"

    thread = threading.Thread(target=_run_training, daemon=True)
    thread.start()

    return {"run_id": run_id, "message": "Training started"}


@router.get("/{run_id}")
def get_training_status(run_id: str):
    """Get training run status."""
    # Check active tasks first
    if run_id in _training_tasks:
        return _training_tasks[run_id]

    # Check saved runs
    run_dir = RUNS_DIR / run_id
    if run_dir.exists():
        return _load_run_meta(run_dir)

    raise HTTPException(404, "Training run not found")


@router.post("/{run_id}/stop")
def stop_training(run_id: str):
    """Stop a running training job."""
    if run_id not in _training_tasks:
        raise HTTPException(404, "No active training found with this ID")

    task = _training_tasks[run_id]
    if task["status"] != "training":
        raise HTTPException(400, "Training is not running")

    try:
        from ..models.factory import create_trainer
        trainer = create_trainer(task["model_type"], task["model_name"])
        trainer.stop()
    except Exception:
        pass

    task["status"] = "stopped"
    task["message"] = "Training stopped by user"

    run_dir = RUNS_DIR / run_id
    if run_dir.exists():
        meta = _load_run_meta(run_dir)
        meta["status"] = "stopped"
        meta["completed_at"] = datetime.now().isoformat()
        _save_run_meta(run_dir, meta)

    return {"message": "Training stopped"}


@router.delete("/{run_id}")
def delete_training_run(run_id: str):
    """Delete a training run and its data."""
    import shutil

    run_dir = RUNS_DIR / run_id
    if not run_dir.exists():
        raise HTTPException(404, "Training run not found")

    if run_id in _training_tasks and _training_tasks[run_id]["status"] == "training":
        raise HTTPException(400, "Cannot delete a running training job. Stop it first.")

    shutil.rmtree(run_dir)
    if run_id in _training_tasks:
        del _training_tasks[run_id]

    return {"message": "Training run deleted"}


@router.get("/{run_id}/logs")
def get_training_logs(run_id: str):
    """Get training logs if available."""
    run_dir = RUNS_DIR / run_id
    if not run_dir.exists():
        raise HTTPException(404, "Training run not found")

    # Look for results in the run directory
    logs = []
    model_name_dir = None
    for d in run_dir.iterdir():
        if d.is_dir() and not d.name.startswith("."):
            model_name_dir = d
            break

    if model_name_dir:
        # Check for results.csv
        results_csv = model_name_dir / "results.csv"
        if results_csv.exists():
            lines = results_csv.read_text().strip().split("\n")
            headers = lines[0].split(",")
            for line in lines[1:]:
                values = line.split(",")
                logs.append(dict(zip([h.strip() for h in headers], [v.strip() for v in values])))

    return {"logs": logs}
