"""
api_server.py — FastAPI /metrics endpoint for NeuroScan AI
─────────────────────────────────────────────────────────────
Start server:
    uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    GET  /metrics            → JSON metrics + image paths
    GET  /metrics/{model}    → JSON metrics for one model
    GET  /images/{filename}  → serve a results image
    GET  /health             → liveness probe

Install extras:
    pip install fastapi uvicorn[standard] aiofiles
"""

import os
import json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Paths ─────────────────────────────────────────────────────────────────────
RESULTS_DIR  = Path("results")
REPORT_FILE  = RESULTS_DIR / "metrics_report.json"

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="NeuroScan AI — Metrics API",
    description=(
        "REST API exposing evaluation metrics and visualisation paths "
        "for the NeuroScan multi-modal Parkinson's Detection System."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _load_report() -> dict:
    if not REPORT_FILE.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "metrics_report.json not found. "
                "Run training first: `python train.py`"
            ),
        )
    with open(REPORT_FILE) as f:
        return json.load(f)


def _clean_metrics(m: dict) -> dict:
    """Strip internal _roc_fpr / _roc_tpr arrays from public response."""
    return {k: v for k, v in m.items() if not k.startswith("_")}


# ═══════════════════════════════════════════════════════════════════════════════
#  Response models
# ═══════════════════════════════════════════════════════════════════════════════

class ImagePaths(BaseModel):
    confusion_matrix:  Optional[str]
    accuracy_graph:    Optional[str]
    loss_graph:        Optional[str]
    metrics_dashboard: Optional[str]


class MetricsResponse(BaseModel):
    generated_at: str
    models:       dict
    image_paths:  ImagePaths


# ═══════════════════════════════════════════════════════════════════════════════
#  Routes
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/health", tags=["status"])
def health():
    """Liveness probe."""
    report_ready = REPORT_FILE.exists()
    images = {}
    if report_ready:
        try:
            rpt = _load_report()
            images = {k: Path(v).exists() if v else False
                      for k, v in rpt.get("image_paths", {}).items()}
        except Exception:
            pass
    return {"status": "ok", "report_ready": report_ready, "images": images}


@app.get("/metrics", response_class=JSONResponse, tags=["metrics"],
         summary="All model metrics + image paths")
def get_all_metrics():
    """
    Returns scalar metrics (accuracy, precision, recall, F1, specificity,
    confusion matrix) for every trained model, plus absolute paths to the
    generated PNG files.
    """
    report = _load_report()
    return {
        "generated_at": report["generated_at"],
        "models": {
            name: _clean_metrics(m)
            for name, m in report["models"].items()
        },
        "image_paths": report.get("image_paths", {}),
    }


@app.get("/metrics/{model_name}", response_class=JSONResponse, tags=["metrics"],
         summary="Metrics for a single model")
def get_model_metrics(model_name: str):
    """
    model_name: one of  mri | video | scribble | fusion
    Returns the full metric dict for that model.
    """
    report = _load_report()
    models = report.get("models", {})
    if model_name not in models:
        available = list(models.keys())
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found. Available: {available}",
        )
    return _clean_metrics(models[model_name])


@app.get("/images/{filename}", tags=["images"],
         summary="Serve a results PNG image")
def get_image(filename: str):
    """
    filename: confusion_matrix.png | accuracy_graph.png |
              loss_graph.png       | metrics_dashboard.png
    """
    # Sanitise: only allow .png files inside results/
    if not filename.endswith(".png") or "/" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename.")
    path = RESULTS_DIR / filename
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"{filename} not found. Run training to generate it.",
        )
    return FileResponse(str(path), media_type="image/png")


@app.get("/summary", response_class=JSONResponse, tags=["metrics"],
         summary="Compact leaderboard across all models")
def get_summary():
    """
    Returns a concise leaderboard sorted by F1-score descending.
    """
    report = _load_report()
    rows = []
    for name, m in report["models"].items():
        rows.append({
            "model":       name,
            "label":       {"mri": "MRI", "video": "Tremor Video",
                            "scribble": "Scribble Test",
                            "fusion": "Multi-Modal Fusion"}.get(name, name),
            "accuracy":    m.get("accuracy"),
            "precision":   m.get("precision"),
            "recall":      m.get("recall"),
            "f1_score":    m.get("f1_score"),
            "specificity": m.get("specificity"),
            "roc_auc":     m.get("roc_auc"),
            "n_samples":   m.get("n_samples"),
        })
    rows.sort(key=lambda r: r["f1_score"] or 0, reverse=True)
    return {"leaderboard": rows, "generated_at": report["generated_at"]}


# ═══════════════════════════════════════════════════════════════════════════════
#  Dev entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)
