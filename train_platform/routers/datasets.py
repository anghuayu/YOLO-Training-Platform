"""
Dataset management API routes.
Includes: CRUD, image upload, video-to-frames import, version management.
"""
import json
import shutil
import uuid
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, List

import cv2
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ..config import DATASETS_DIR, IMAGE_EXTENSIONS, VIDEO_EXTENSIONS, VIDEO_IMPORT_DEFAULTS

router = APIRouter(prefix="/api/datasets", tags=["datasets"])


# --- Schemas ---
class DatasetCreate(BaseModel):
    name: str
    classes: List[str] = []
    description: str = ""


class DatasetUpdate(BaseModel):
    name: Optional[str] = None
    classes: Optional[List[str]] = None
    description: Optional[str] = None


class DatasetInfo(BaseModel):
    id: str
    name: str
    classes: List[str]
    description: str
    image_count: int
    annotated_count: int
    created_at: str
    updated_at: str


class BatchDeleteRequest(BaseModel):
    filenames: List[str]


# --- Helpers ---
def _load_meta(ds_dir: Path) -> dict:
    meta_file = ds_dir / "meta.json"
    if meta_file.exists():
        return json.loads(meta_file.read_text(encoding="utf-8"))
    return {}


def _save_meta(ds_dir: Path, meta: dict):
    (ds_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _count_images(ds_dir: Path) -> int:
    img_dir = ds_dir / "images"
    if not img_dir.exists():
        return 0
    return sum(1 for f in img_dir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS)


def _count_annotated(ds_dir: Path) -> int:
    lbl_dir = ds_dir / "labels"
    if not lbl_dir.exists():
        return 0
    img_dir = ds_dir / "images"
    count = 0
    for lbl in lbl_dir.glob("*.txt"):
        img_stem = lbl.stem
        # Check if corresponding image exists
        for ext in IMAGE_EXTENSIONS:
            if (img_dir / f"{img_stem}{ext}").exists():
                count += 1
                break
    return count


# --- Routes ---
@router.get("")
def list_datasets():
    """List all datasets."""
    datasets = []
    if not DATASETS_DIR.exists():
        return datasets
    for ds_dir in sorted(DATASETS_DIR.iterdir()):
        if ds_dir.is_dir():
            meta = _load_meta(ds_dir)
            if not meta:
                continue
            datasets.append(
                {
                    "id": ds_dir.name,
                    "name": meta.get("name", ds_dir.name),
                    "classes": meta.get("classes", []),
                    "description": meta.get("description", ""),
                    "total_images": _count_images(ds_dir),
                    "annotated_images": _count_annotated(ds_dir),
                    "version_count": len(meta.get("versions", [])),
                    "created_at": meta.get("created_at", ""),
                    "updated_at": meta.get("updated_at", ""),
                }
            )
    return datasets


@router.post("")
def create_dataset(req: DatasetCreate):
    """Create a new empty dataset."""
    ds_id = uuid.uuid4().hex[:12]
    ds_dir = DATASETS_DIR / ds_id
    ds_dir.mkdir(parents=True)
    (ds_dir / "images").mkdir()
    (ds_dir / "labels").mkdir()

    meta = {
        "id": ds_id,
        "name": req.name,
        "classes": req.classes,
        "description": req.description,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
    }
    _save_meta(ds_dir, meta)
    return {"id": ds_id, "name": req.name, "message": "Dataset created"}


@router.get("/{dataset_id}")
def get_dataset(dataset_id: str):
    """Get dataset details."""
    ds_dir = DATASETS_DIR / dataset_id
    if not ds_dir.exists():
        raise HTTPException(404, "Dataset not found")
    meta = _load_meta(ds_dir)
    if not meta:
        raise HTTPException(404, "Dataset metadata not found")
    return {
        "id": ds_dir.name,
        "name": meta.get("name", ""),
        "classes": meta.get("classes", []),
        "description": meta.get("description", ""),
        "image_count": _count_images(ds_dir),
        "annotated_count": _count_annotated(ds_dir),
        "created_at": meta.get("created_at", ""),
        "updated_at": meta.get("updated_at", ""),
    }


@router.put("/{dataset_id}")
def update_dataset(dataset_id: str, req: DatasetUpdate):
    """Update dataset metadata."""
    ds_dir = DATASETS_DIR / dataset_id
    if not ds_dir.exists():
        raise HTTPException(404, "Dataset not found")
    meta = _load_meta(ds_dir)
    if req.name is not None:
        meta["name"] = req.name
    if req.classes is not None:
        meta["classes"] = req.classes
    if req.description is not None:
        meta["description"] = req.description
    meta["updated_at"] = datetime.now().isoformat()
    _save_meta(ds_dir, meta)
    return {"message": "Dataset updated"}


@router.delete("/{dataset_id}")
def delete_dataset(dataset_id: str):
    """Delete a dataset and all its data."""
    ds_dir = DATASETS_DIR / dataset_id
    if not ds_dir.exists():
        raise HTTPException(404, "Dataset not found")
    shutil.rmtree(ds_dir)
    return {"message": "Dataset deleted"}


@router.post("/{dataset_id}/images")
async def upload_images(
    dataset_id: str,
    files: List[UploadFile] = File(...),
):
    """Upload images to a dataset."""
    ds_dir = DATASETS_DIR / dataset_id
    if not ds_dir.exists():
        raise HTTPException(404, "Dataset not found")

    img_dir = ds_dir / "images"
    img_dir.mkdir(exist_ok=True)

    uploaded = []
    for f in files:
        # Strip directory components (folder upload includes relative paths like "images/0001.jpg")
        raw_name = f.filename.replace("\\", "/")
        safe_name = raw_name.rsplit("/", 1)[-1]
        if Path(safe_name).suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        # Avoid name collision
        dest_name = safe_name
        dest = img_dir / dest_name
        if dest.exists():
            stem = Path(safe_name).stem
            ext = Path(safe_name).suffix
            dest_name = f"{stem}_{uuid.uuid4().hex[:6]}{ext}"
            dest = img_dir / dest_name

        content = await f.read()
        dest.write_bytes(content)
        uploaded.append(dest_name)

    # Update timestamp
    meta = _load_meta(ds_dir)
    meta["updated_at"] = datetime.now().isoformat()
    _save_meta(ds_dir, meta)

    return {"uploaded": len(uploaded), "files": uploaded}


@router.get("/{dataset_id}/images")
def list_images(dataset_id: str):
    """List all images in a dataset with annotation status."""
    ds_dir = DATASETS_DIR / dataset_id
    if not ds_dir.exists():
        raise HTTPException(404, "Dataset not found")

    img_dir = ds_dir / "images"
    lbl_dir = ds_dir / "labels"
    if not img_dir.exists():
        return []

    images = []
    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        has_label = (lbl_dir / f"{img_path.stem}.txt").exists()
        images.append(
            {
                "filename": img_path.name,
                "path": f"/api/datasets/{dataset_id}/images/{img_path.name}",
                "has_annotation": has_label,
                "size": img_path.stat().st_size,
            }
        )
    return images


@router.get("/{dataset_id}/images/{filename}")
def get_image(dataset_id: str, filename: str):
    """Get a specific image file."""
    img_path = DATASETS_DIR / dataset_id / "images" / filename
    if not img_path.exists():
        raise HTTPException(404, "Image not found")
    return FileResponse(img_path)


@router.delete("/{dataset_id}/images/{filename}")
def delete_image(dataset_id: str, filename: str):
    """Delete an image and its annotation."""
    ds_dir = DATASETS_DIR / dataset_id
    img_path = ds_dir / "images" / filename
    lbl_path = ds_dir / "labels" / f"{Path(filename).stem}.txt"

    if not img_path.exists():
        raise HTTPException(404, "Image not found")

    img_path.unlink()
    if lbl_path.exists():
        lbl_path.unlink()

    return {"message": "Image deleted"}


@router.post("/{dataset_id}/images/batch-delete")
def batch_delete_images(dataset_id: str, req: BatchDeleteRequest):
    """Batch delete multiple images and their annotations."""
    ds_dir = DATASETS_DIR / dataset_id
    if not ds_dir.exists():
        raise HTTPException(404, "Dataset not found")

    img_dir = ds_dir / "images"
    lbl_dir = ds_dir / "labels"

    deleted = 0
    not_found = 0

    for filename in req.filenames:
        img_path = img_dir / filename
        if img_path.exists():
            img_path.unlink()
            lbl_path = lbl_dir / f"{Path(filename).stem}.txt"
            if lbl_path.exists():
                lbl_path.unlink()
            deleted += 1
        else:
            not_found += 1

    # Update timestamp
    meta = _load_meta(ds_dir)
    meta["updated_at"] = datetime.now().isoformat()
    _save_meta(ds_dir, meta)

    return {"deleted": deleted, "not_found": not_found}


@router.get("/{dataset_id}/export-yolo")
def export_yolo_format(dataset_id: str):
    """Export dataset in YOLO format as a zip file."""
    import zipfile
    import io

    ds_dir = DATASETS_DIR / dataset_id
    if not ds_dir.exists():
        raise HTTPException(404, "Dataset not found")

    meta = _load_meta(ds_dir)
    classes = meta.get("classes", [])

    # Create data.yaml
    yaml_content = f"path: {ds_dir}\ntrain: images\nval: images\n"
    yaml_content += f"nc: {len(classes)}\n"
    yaml_content += f"names: {classes}\n"

    # Create zip
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("data.yaml", yaml_content)
        img_dir = ds_dir / "images"
        lbl_dir = ds_dir / "labels"
        if img_dir.exists():
            for img in img_dir.iterdir():
                if img.suffix.lower() in IMAGE_EXTENSIONS:
                    zf.write(img, f"images/{img.name}")
                    lbl = lbl_dir / f"{img.stem}.txt"
                    if lbl.exists():
                        zf.write(lbl, f"labels/{lbl.name}")

    zip_buffer.seek(0)
    from fastapi.responses import StreamingResponse

    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={meta.get('name', dataset_id)}_yolo.zip"},
    )


# ============================================
# Video Import - Extract frames from video
# ============================================

# In-memory task tracking for video extraction
_video_tasks: dict = {}


@router.post("/{dataset_id}/import-video")
async def import_video(
    dataset_id: str,
    file: UploadFile = File(...),
    interval_sec: float = Form(default=1.0),
    jpeg_quality: int = Form(default=95),
    max_frames: int = Form(default=500),
):
    """
    Upload a video and extract frames at a fixed interval (seconds).
    Frames are saved as JPG into the dataset's images/ folder.
    The original video is stored in videos/ for reference.
    """
    ds_dir = DATASETS_DIR / dataset_id
    if not ds_dir.exists():
        raise HTTPException(404, "Dataset not found")

    # Validate extension
    suffix = Path(file.filename).suffix.lower()
    if suffix not in VIDEO_EXTENSIONS:
        raise HTTPException(
            400,
            f"Unsupported video format '{suffix}'. Supported: {', '.join(sorted(VIDEO_EXTENSIONS))}",
        )

    # Save original video
    video_dir = ds_dir / "videos"
    video_dir.mkdir(exist_ok=True)
    video_path = video_dir / file.filename
    # Avoid collision
    if video_path.exists():
        stem = Path(file.filename).stem
        video_path = video_dir / f"{stem}_{uuid.uuid4().hex[:6]}{suffix}"

    content = await file.read()
    video_path.write_bytes(content)

    # Extract frames
    img_dir = ds_dir / "images"
    img_dir.mkdir(exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise HTTPException(400, "Cannot open video file. It may be corrupted.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 0

    # Calculate frame step
    frame_step = max(1, int(fps * interval_sec))
    jpeg_params = [cv2.IMWRITE_JPEG_QUALITY, max(1, min(100, jpeg_quality))]

    # Generate a prefix for this batch
    batch_id = uuid.uuid4().hex[:8]
    video_stem = Path(file.filename).stem

    extracted = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_step == 0:
            # Save as JPG
            out_name = f"{video_stem}_{batch_id}_{extracted:05d}.jpg"
            out_path = img_dir / out_name
            cv2.imwrite(str(out_path), frame, jpeg_params)
            extracted += 1

            if extracted >= max_frames:
                break

        frame_idx += 1

    cap.release()

    # Update dataset meta
    meta = _load_meta(ds_dir)
    meta["updated_at"] = datetime.now().isoformat()
    # Track video imports
    if "video_imports" not in meta:
        meta["video_imports"] = []
    meta["video_imports"].append({
        "filename": video_path.name,
        "interval_sec": interval_sec,
        "frames_extracted": extracted,
        "duration_sec": round(duration_sec, 2),
        "fps": round(fps, 2),
        "imported_at": datetime.now().isoformat(),
    })
    _save_meta(ds_dir, meta)

    return {
        "message": f"Extracted {extracted} frames from video",
        "video_file": video_path.name,
        "frames_extracted": extracted,
        "video_duration_sec": round(duration_sec, 2),
        "video_fps": round(fps, 2),
        "interval_sec": interval_sec,
    }


@router.get("/{dataset_id}/videos")
def list_videos(dataset_id: str):
    """List all imported videos for a dataset."""
    ds_dir = DATASETS_DIR / dataset_id
    if not ds_dir.exists():
        raise HTTPException(404, "Dataset not found")

    video_dir = ds_dir / "videos"
    if not video_dir.exists():
        return []

    videos = []
    for vf in sorted(video_dir.iterdir()):
        if vf.suffix.lower() in VIDEO_EXTENSIONS:
            videos.append({
                "filename": vf.name,
                "size": vf.stat().st_size,
                "path": f"/api/datasets/{dataset_id}/videos/{vf.name}",
            })
    return videos


@router.get("/{dataset_id}/videos/{filename}")
def get_video(dataset_id: str, filename: str):
    """Serve a video file."""
    video_path = DATASETS_DIR / dataset_id / "videos" / filename
    if not video_path.exists():
        raise HTTPException(404, "Video not found")
    return FileResponse(video_path, media_type="video/mp4")


# ============================================
# Version Management
# ============================================

@router.post("/{dataset_id}/versions")
def create_version(dataset_id: str, name: str = Form(default=""), note: str = Form(default="")):
    """
    Create a version snapshot of the current dataset (images + labels).
    Versions are stored under versions/v{N}_{timestamp}/.
    """
    ds_dir = DATASETS_DIR / dataset_id
    if not ds_dir.exists():
        raise HTTPException(404, "Dataset not found")

    meta = _load_meta(ds_dir)
    versions = meta.get("versions", [])

    # Determine next version number
    next_num = len(versions) + 1
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_id = f"v{next_num}_{timestamp}"
    version_name = name.strip() if name.strip() else f"Version {next_num}"

    # Create version directory and copy images + labels
    ver_dir = ds_dir / "versions" / version_id
    ver_img_dir = ver_dir / "images"
    ver_lbl_dir = ver_dir / "labels"
    ver_img_dir.mkdir(parents=True)
    ver_lbl_dir.mkdir(parents=True)

    img_dir = ds_dir / "images"
    lbl_dir = ds_dir / "labels"

    img_count = 0
    lbl_count = 0

    if img_dir.exists():
        for img in img_dir.iterdir():
            if img.suffix.lower() in IMAGE_EXTENSIONS:
                shutil.copy2(img, ver_img_dir / img.name)
                img_count += 1

    if lbl_dir.exists():
        for lbl in lbl_dir.glob("*.txt"):
            shutil.copy2(lbl, ver_lbl_dir / lbl.name)
            lbl_count += 1

    # Record version metadata
    version_info = {
        "id": version_id,
        "name": version_name,
        "note": note.strip(),
        "image_count": img_count,
        "label_count": lbl_count,
        "created_at": datetime.now().isoformat(),
    }
    versions.append(version_info)
    meta["versions"] = versions
    meta["updated_at"] = datetime.now().isoformat()
    _save_meta(ds_dir, meta)

    return {
        "message": f"Version '{version_name}' created",
        "version": version_info,
    }


@router.get("/{dataset_id}/versions")
def list_versions(dataset_id: str):
    """List all versions of a dataset."""
    ds_dir = DATASETS_DIR / dataset_id
    if not ds_dir.exists():
        raise HTTPException(404, "Dataset not found")

    meta = _load_meta(ds_dir)
    versions = meta.get("versions", [])
    # Return in reverse order (newest first)
    return list(reversed(versions))


@router.post("/{dataset_id}/versions/{version_id}/restore")
def restore_version(dataset_id: str, version_id: str):
    """
    Restore a version: replace current images/ and labels/ with the version's snapshot.
    """
    ds_dir = DATASETS_DIR / dataset_id
    if not ds_dir.exists():
        raise HTTPException(404, "Dataset not found")

    ver_dir = ds_dir / "versions" / version_id
    if not ver_dir.exists():
        raise HTTPException(404, "Version not found")

    img_dir = ds_dir / "images"
    lbl_dir = ds_dir / "labels"

    # Clear current images and labels
    if img_dir.exists():
        shutil.rmtree(img_dir)
    if lbl_dir.exists():
        shutil.rmtree(lbl_dir)

    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)

    # Copy version data back
    ver_img_dir = ver_dir / "images"
    ver_lbl_dir = ver_dir / "labels"

    restored_imgs = 0
    restored_lbls = 0

    if ver_img_dir.exists():
        for img in ver_img_dir.iterdir():
            shutil.copy2(img, img_dir / img.name)
            restored_imgs += 1

    if ver_lbl_dir.exists():
        for lbl in ver_lbl_dir.iterdir():
            shutil.copy2(lbl, lbl_dir / lbl.name)
            restored_lbls += 1

    # Update meta
    meta = _load_meta(ds_dir)
    meta["updated_at"] = datetime.now().isoformat()
    _save_meta(ds_dir, meta)

    return {
        "message": f"Restored version '{version_id}'",
        "restored_images": restored_imgs,
        "restored_labels": restored_lbls,
    }


@router.delete("/{dataset_id}/versions/{version_id}")
def delete_version(dataset_id: str, version_id: str):
    """Delete a specific version snapshot."""
    ds_dir = DATASETS_DIR / dataset_id
    if not ds_dir.exists():
        raise HTTPException(404, "Dataset not found")

    ver_dir = ds_dir / "versions" / version_id
    if not ver_dir.exists():
        raise HTTPException(404, "Version not found")

    # Remove version directory
    shutil.rmtree(ver_dir)

    # Remove from meta
    meta = _load_meta(ds_dir)
    versions = meta.get("versions", [])
    meta["versions"] = [v for v in versions if v.get("id") != version_id]
    meta["updated_at"] = datetime.now().isoformat()
    _save_meta(ds_dir, meta)

    return {"message": f"Version '{version_id}' deleted"}
