"""
Annotation management API routes, including smart auto-annotation.
"""
import json
import uuid
import threading
from pathlib import Path
from typing import Optional, List

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ..config import DATASETS_DIR, SMART_ANNOTATION, RUNS_DIR

router = APIRouter(prefix="/api/annotations", tags=["annotations"])


# --- Schemas ---
class BBox(BaseModel):
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float


class AnnotationSave(BaseModel):
    image_name: str
    boxes: List[BBox]


class SmartAnnotationRequest(BaseModel):
    model_type: str = "yolov8"
    pretrained: str = ""
    epochs: int = 50
    imgsz: int = 640
    confidence: float = 0.25


# --- Helpers ---
def _read_labels(ds_dir: Path, image_stem: str) -> List[BBox]:
    lbl_file = ds_dir / "labels" / f"{image_stem}.txt"
    boxes = []
    if lbl_file.exists():
        for line in lbl_file.read_text().strip().split("\n"):
            if not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) >= 5:
                boxes.append(
                    BBox(
                        class_id=int(parts[0]),
                        x_center=float(parts[1]),
                        y_center=float(parts[2]),
                        width=float(parts[3]),
                        height=float(parts[4]),
                    )
                )
    return boxes


def _write_labels(ds_dir: Path, image_stem: str, boxes: List[BBox]):
    lbl_file = ds_dir / "labels" / f"{image_stem}.txt"
    lines = []
    for b in boxes:
        lines.append(f"{b.class_id} {b.x_center} {b.y_center} {b.width} {b.height}")
    lbl_file.write_text("\n".join(lines), encoding="utf-8")


def _generate_data_yaml(ds_dir: Path) -> str:
    """Generate a data.yaml for training from dataset directory."""
    meta_file = ds_dir / "meta.json"
    meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {}
    classes = meta.get("classes", [])
    yaml_path = ds_dir / "data.yaml"
    content = f"path: {ds_dir}\ntrain: images\nval: images\n"
    content += f"nc: {len(classes)}\n"
    content += f"names: {list(classes)}\n"
    yaml_path.write_text(content, encoding="utf-8")
    return str(yaml_path)


# --- Routes ---
@router.get("/{dataset_id}")
def list_annotations(dataset_id: str):
    """List all annotations for a dataset."""
    ds_dir = DATASETS_DIR / dataset_id
    if not ds_dir.exists():
        raise HTTPException(404, "Dataset not found")

    from ..config import IMAGE_EXTENSIONS

    img_dir = ds_dir / "images"
    if not img_dir.exists():
        return []

    results = []
    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        boxes = _read_labels(ds_dir, img_path.stem)
        results.append(
            {
                "image": img_path.name,
                "image_path": f"/api/datasets/{dataset_id}/images/{img_path.name}",
                "boxes": [b.model_dump() for b in boxes],
                "annotated": len(boxes) > 0,
            }
        )
    return results


@router.get("/{dataset_id}/{image_name}")
def get_annotation(dataset_id: str, image_name: str):
    """Get annotations for a specific image."""
    ds_dir = DATASETS_DIR / dataset_id
    if not ds_dir.exists():
        raise HTTPException(404, "Dataset not found")

    stem = Path(image_name).stem
    boxes = _read_labels(ds_dir, stem)
    return {
        "image": image_name,
        "boxes": [b.model_dump() for b in boxes],
    }


@router.put("/{dataset_id}/{image_name}")
def save_annotation(dataset_id: str, image_name: str, req: AnnotationSave):
    """Save annotations for a specific image (overwrite)."""
    ds_dir = DATASETS_DIR / dataset_id
    if not ds_dir.exists():
        raise HTTPException(404, "Dataset not found")

    stem = Path(image_name).stem
    _write_labels(ds_dir, stem, req.boxes)

    # Update dataset timestamp
    meta_file = ds_dir / "meta.json"
    if meta_file.exists():
        from datetime import datetime
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        meta["updated_at"] = datetime.now().isoformat()
        meta_file.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    return {"message": "Annotation saved", "box_count": len(req.boxes)}


@router.delete("/{dataset_id}/{image_name}")
def delete_annotation(dataset_id: str, image_name: str):
    """Delete annotations for a specific image."""
    ds_dir = DATASETS_DIR / dataset_id
    if not ds_dir.exists():
        raise HTTPException(404, "Dataset not found")

    lbl_file = ds_dir / "labels" / f"{Path(image_name).stem}.txt"
    if lbl_file.exists():
        lbl_file.unlink()
    return {"message": "Annotation deleted"}


# --- Smart Annotation ---
_smart_tasks = {}


@router.post("/{dataset_id}/smart-annotate")
def start_smart_annotation(dataset_id: str, req: SmartAnnotationRequest):
    """
    Start smart annotation process:
    1. Train a temporary model on existing annotations
    2. Use the model to predict on unannotated images
    """
    ds_dir = DATASETS_DIR / dataset_id
    if not ds_dir.exists():
        raise HTTPException(404, "Dataset not found")

    # Check minimum annotated images
    from ..config import IMAGE_EXTENSIONS
    img_dir = ds_dir / "images"
    lbl_dir = ds_dir / "labels"
    annotated_count = 0
    unannotated = []
    for img in img_dir.iterdir():
        if img.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if (lbl_dir / f"{img.stem}.txt").exists():
            annotated_count += 1
        else:
            unannotated.append(img.name)

    if annotated_count < SMART_ANNOTATION["min_images"]:
        raise HTTPException(
            400,
            f"Need at least {SMART_ANNOTATION['min_images']} annotated images, "
            f"but only {annotated_count} found. Please annotate more images first.",
        )

    if not unannotated:
        raise HTTPException(400, "All images are already annotated.")

    task_id = uuid.uuid4().hex[:12]
    _smart_tasks[task_id] = {
        "status": "training",
        "progress": 0,
        "dataset_id": dataset_id,
        "total_images": len(unannotated),
        "processed": 0,
        "predicted_count": 0,
        "message": "Starting smart annotation...",
    }

    def _run_smart_annotation():
        try:
            from ..models.factory import create_trainer

            # Determine pretrained model to use for prediction
            pretrained = req.pretrained if req.pretrained and Path(req.pretrained).exists() else ""

            # Step 1: Generate data.yaml with proper train/val split
            meta_file = ds_dir / "meta.json"
            meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {}
            classes = meta.get("classes", [])

            # Create train/val split to prevent overfitting
            img_dir_split = ds_dir / "images"
            all_images = sorted([f for f in img_dir_split.iterdir()
                                 if f.suffix.lower() in IMAGE_EXTENSIONS])
            split_idx = max(1, int(len(all_images) * 0.8))
            train_dir = ds_dir / "images_train"
            val_dir = ds_dir / "images_val"
            train_dir.mkdir(exist_ok=True)
            val_dir.mkdir(exist_ok=True)

            import shutil
            for img in all_images[:split_idx]:
                dst = train_dir / img.name
                if not dst.exists():
                    shutil.copy2(str(img), str(dst))
            for img in all_images[split_idx:]:
                dst = val_dir / img.name
                if not dst.exists():
                    shutil.copy2(str(img), str(dst))

            # Also copy corresponding labels
            lbl_dir = ds_dir / "labels"
            train_lbl = ds_dir / "labels_train"
            val_lbl = ds_dir / "labels_val"
            train_lbl.mkdir(exist_ok=True)
            val_lbl.mkdir(exist_ok=True)
            for img in all_images[:split_idx]:
                lbl = lbl_dir / f"{img.stem}.txt"
                if lbl.exists():
                    shutil.copy2(str(lbl), str(train_lbl / lbl.name))
            for img in all_images[split_idx:]:
                lbl = lbl_dir / f"{img.stem}.txt"
                if lbl.exists():
                    shutil.copy2(str(lbl), str(val_lbl / lbl.name))

            # Write data.yaml with split paths
            yaml_path = ds_dir / "data.yaml"
            yaml_content = (
                f"path: {ds_dir}\n"
                f"train: images_train\n"
                f"val: images_val\n"
                f"nc: {len(classes)}\n"
                f"names: {list(classes)}\n"
            )
            yaml_path.write_text(yaml_content, encoding="utf-8")
            data_yaml = str(yaml_path)

            # Step 2: Train temporary model with proper split
            task_name = f"smart_{task_id}"
            output_dir = str(RUNS_DIR / "smart_annotation")
            trainer = create_trainer(req.model_type, task_name)

            _smart_tasks[task_id]["message"] = "Training temporary model..."
            result = trainer.train(
                data_yaml=data_yaml,
                epochs=req.epochs,
                imgsz=req.imgsz,
                batch=8,
                output_dir=output_dir,
                pretrained=pretrained,
                device="0",
            )

            # Step 3: Predict using the PRETRAINED model (not fine-tuned)
            # Fine-tuned models overfit on small datasets and fail to generalize.
            # The pretrained model has much better generalization ability.
            predict_model = pretrained if pretrained else "yolov8n.pt"
            _smart_tasks[task_id]["status"] = "predicting"
            _smart_tasks[task_id]["message"] = "Predicting on unannotated images..."

            # Load pretrained model once
            from ultralytics import YOLO
            pred_model = YOLO(predict_model)

            # Build mapping: pretrained model class name -> user dataset class id
            # This filters out irrelevant detections (e.g., skateboard when user only has "person")
            model_names = pred_model.names  # {0: 'person', 1: 'bicycle', ...}
            user_classes_lower = [c.lower() for c in classes]
            # Map: pretrained_class_id -> user_class_id
            class_mapping = {}
            for pre_id, pre_name in model_names.items():
                if pre_name.lower() in user_classes_lower:
                    user_idx = user_classes_lower.index(pre_name.lower())
                    class_mapping[pre_id] = user_idx
            # If no matches at all, fall back to mapping everything to class 0
            if not class_mapping and classes:
                class_mapping = {k: 0 for k in model_names}

            predicted_count = 0
            error_count = 0
            error_messages = []
            for i, img_name in enumerate(unannotated):
                img_path = str(img_dir / img_name)
                try:
                    results = pred_model(img_path, conf=req.confidence, imgsz=req.imgsz, verbose=False)
                    boxes = []
                    if results and len(results) > 0:
                        r = results[0]
                        if r.boxes is not None and len(r.boxes) > 0:
                            import cv2
                            img_cv = cv2.imread(img_path)
                            if img_cv is None:
                                error_count += 1
                                if len(error_messages) < 3:
                                    error_messages.append(f"cv2.imread failed: {img_name}")
                            else:
                                h, w = img_cv.shape[:2]
                                for box in r.boxes:
                                    xyxy = box.xyxy[0].tolist()
                                    det_cls = int(box.cls[0])
                                    # Only keep detections that match user's classes
                                    if det_cls not in class_mapping:
                                        continue
                                    user_cls_id = class_mapping[det_cls]
                                    x1, y1, x2, y2 = xyxy
                                    x_center = ((x1 + x2) / 2) / w
                                    y_center = ((y1 + y2) / 2) / h
                                    bw = (x2 - x1) / w
                                    bh = (y2 - y1) / h
                                    boxes.append(
                                        BBox(
                                            class_id=user_cls_id,
                                            x_center=round(x_center, 6),
                                            y_center=round(y_center, 6),
                                            width=round(bw, 6),
                                            height=round(bh, 6),
                                        )
                                    )

                    if boxes:
                        _write_labels(ds_dir, Path(img_name).stem, boxes)
                        predicted_count += 1

                    _smart_tasks[task_id]["processed"] = i + 1
                    _smart_tasks[task_id]["predicted_count"] = predicted_count
                    _smart_tasks[task_id]["progress"] = int(
                        (i + 1) / len(unannotated) * 100
                    )
                except Exception as e:
                    error_count += 1
                    if len(error_messages) < 3:
                        error_messages.append(f"{img_name}: {str(e)}")
                    continue

            # Cleanup temporary split directories
            for d in [train_dir, val_dir, train_lbl, val_lbl]:
                if d.exists():
                    shutil.rmtree(d, ignore_errors=True)

            # Clean up ultralytics cache files
            for cache_file in ds_dir.glob("*.cache"):
                try:
                    cache_file.unlink()
                except Exception:
                    pass

            # Restore data.yaml to point to original images/labels directories
            _generate_data_yaml(ds_dir)

            _smart_tasks[task_id]["error_count"] = error_count
            _smart_tasks[task_id]["error_messages"] = error_messages
            _smart_tasks[task_id]["status"] = "completed"
            _smart_tasks[task_id]["message"] = (
                f"Smart annotation complete! Predicted {predicted_count}/{len(unannotated)} images."
            )
            if error_count > 0:
                _smart_tasks[task_id]["message"] += f" ({error_count} errors)"
            _smart_tasks[task_id]["progress"] = 100

        except Exception as e:
            _smart_tasks[task_id]["status"] = "error"
            _smart_tasks[task_id]["message"] = f"Error: {str(e)}"

    thread = threading.Thread(target=_run_smart_annotation, daemon=True)
    thread.start()

    return {"task_id": task_id, "message": "Smart annotation started"}


@router.get("/{dataset_id}/smart-annotate/status")
def get_smart_annotation_status(dataset_id: str, task_id: str = ""):
    """Get smart annotation task status."""
    if task_id and task_id in _smart_tasks:
        return _smart_tasks[task_id]
    # Return latest task for this dataset
    for tid, task in reversed(_smart_tasks.items()):
        if task["dataset_id"] == dataset_id:
            return task
    return {"status": "none", "message": "No smart annotation task found"}
