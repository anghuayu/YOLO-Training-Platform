"""
YOLO Training Platform - Main Entry Point
Run: python main.py
Then open: http://localhost:8080
"""
import socket
import uvicorn
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from train_platform.config import HOST, PORT, BASE_DIR
from train_platform.routers import datasets, annotations, training, models


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown hooks."""
    print("=" * 60)
    print("  YOLO Training Platform")
    print("=" * 60)
    ip = _get_local_ip()
    print(f"  Local:   http://localhost:{PORT}")
    print(f"  Network: http://{ip}:{PORT}")
    print("=" * 60)
    yield


app = FastAPI(
    title="YOLO Training Platform",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS (allow all for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routers
app.include_router(datasets.router)
app.include_router(annotations.router)
app.include_router(training.router)
app.include_router(models.router)

# Serve static frontend files
STATIC_DIR = BASE_DIR.parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# --- Dashboard routes ---
@app.get("/")
async def serve_index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/datasets")
async def serve_datasets():
    return FileResponse(STATIC_DIR / "datasets.html")


@app.get("/annotate/{dataset_id}")
async def serve_annotate(dataset_id: str):
    return FileResponse(STATIC_DIR / "annotate.html")


@app.get("/train")
async def serve_train():
    return FileResponse(STATIC_DIR / "train.html")


@app.get("/models")
async def serve_models():
    return FileResponse(STATIC_DIR / "models.html")


@app.get("/evaluate")
async def serve_evaluate():
    return FileResponse(STATIC_DIR / "evaluate.html")


# --- Dashboard API ---
@app.get("/api/dashboard")
async def dashboard_stats():
    """Aggregate stats for the dashboard."""
    from train_platform.config import DATASETS_DIR, MODELS_DIR, RUNS_DIR, IMAGE_EXTENSIONS
    import json

    # Dataset stats
    ds_count = 0
    total_images = 0
    total_annotated = 0
    if DATASETS_DIR.exists():
        for ds_dir in DATASETS_DIR.iterdir():
            if ds_dir.is_dir() and (ds_dir / "meta.json").exists():
                ds_count += 1
                img_dir = ds_dir / "images"
                lbl_dir = ds_dir / "labels"
                if img_dir.exists():
                    for img in img_dir.iterdir():
                        if img.suffix.lower() in IMAGE_EXTENSIONS:
                            total_images += 1
                            if (lbl_dir / f"{img.stem}.txt").exists():
                                total_annotated += 1

    # Model stats
    model_count = 0
    if MODELS_DIR.exists():
        model_count = sum(
            1 for f in MODELS_DIR.iterdir() if f.suffix in (".pt", ".pth", ".onnx")
        )

    # Training stats
    training_count = 0
    completed_count = 0
    if RUNS_DIR.exists():
        for run_dir in RUNS_DIR.iterdir():
            if run_dir.is_dir() and not run_dir.name.startswith("smart_"):
                meta_file = run_dir / "run_meta.json"
                if meta_file.exists():
                    training_count += 1
                    meta = json.loads(meta_file.read_text(encoding="utf-8"))
                    if meta.get("status") == "completed":
                        completed_count += 1

    # Recent training runs
    recent_runs = []
    if RUNS_DIR.exists():
        runs_data = []
        for run_dir in RUNS_DIR.iterdir():
            if run_dir.is_dir() and not run_dir.name.startswith("smart_"):
                meta_file = run_dir / "run_meta.json"
                if meta_file.exists():
                    meta = json.loads(meta_file.read_text(encoding="utf-8"))
                    runs_data.append(meta)
        runs_data.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        recent_runs = runs_data[:5]

    # Dataset list for chart
    datasets_list = []
    if DATASETS_DIR.exists():
        for ds_dir in sorted(DATASETS_DIR.iterdir()):
            if ds_dir.is_dir() and (ds_dir / "meta.json").exists():
                meta = json.loads((ds_dir / "meta.json").read_text(encoding="utf-8"))
                img_dir = ds_dir / "images"
                lbl_dir = ds_dir / "labels"
                img_count = 0
                ann_count = 0
                if img_dir.exists():
                    for img in img_dir.iterdir():
                        if img.suffix.lower() in IMAGE_EXTENSIONS:
                            img_count += 1
                            if (lbl_dir / f"{img.stem}.txt").exists():
                                ann_count += 1
                datasets_list.append({
                    "id": ds_dir.name,
                    "name": meta.get("name", ds_dir.name),
                    "total_images": img_count,
                    "annotated_images": ann_count,
                    "classes": meta.get("classes", []),
                })

    return {
        "total_datasets": ds_count,
        "total_images": total_images,
        "total_annotated": total_annotated,
        "total_models": model_count,
        "training_runs": training_count,
        "completed_runs": completed_count,
        "recent_training": recent_runs,
        "datasets": datasets_list,
    }


def _get_local_ip():
    """Get local network IP address."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


if __name__ == "__main__":
    uvicorn.run("main:app", host=HOST, port=PORT, reload=True)
