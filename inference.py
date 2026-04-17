"""
inference.py — Inference engine for NeuroScan AI
=================================================
Loads the best saved model, runs predictions, returns structured results.

Usage (programmatic):
    from inference import ParkinsonsInference
    engine = ParkinsonsInference()
    result = engine.predict_image(file_obj, filename="scan.jpg")
    print(result)
"""

import os
import json
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn

from models_arch import get_model
from preprocessing import (
    validate_image_file, validate_video_file,
    preprocess_image_to_tensor, preprocess_video_file,
    ValidationError,
)

SAVE_DIR = "models"
FEATURE_DIM = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─── Model loader ──────────────────────────────────────────────────────────────

def load_best_model(save_dir: str = SAVE_DIR,
                    device: str = DEVICE) -> Tuple[Optional[nn.Module], str, Dict]:
    """
    Load the best model from disk.

    Returns:
        (model_or_None, model_name, info_dict)
        model is None when running in demo mode (no trained weights found).
    """
    info_path = os.path.join(save_dir, "best_model_info.json")
    legacy_info_path = os.path.join(save_dir, "best_model.json")
    info: Dict = {}

    if os.path.exists(info_path):
        try:
            with open(info_path) as f:
                info = json.load(f)
        except Exception:
            info = {}
    elif os.path.exists(legacy_info_path):
        try:
            with open(legacy_info_path) as f:
                info = json.load(f)
        except Exception:
            info = {}

    model_name = info.get("best_model_name", info.get("model", "efficientnetb0"))
    weights_path = os.path.join(save_dir, "best_model.pth")

    if not os.path.exists(weights_path):
        # Fall back to individual model file
        weights_path = os.path.join(save_dir, f"{model_name}_model.pth")
    if not os.path.exists(weights_path):
        weights_path = info.get("best_model_path", info.get("path", weights_path))

    if not os.path.exists(weights_path):
        return None, model_name, info

    try:
        canonical_name = model_name.lower().replace(" ", "")
        alias_map = {
            "customcnn": "custom_cnn",
            "mobilenetv2": "mobilenetv2",
            "efficientnetb0": "efficientnetb0",
        }
        model = get_model(
            alias_map.get(canonical_name, canonical_name),
            pretrained=False,
            freeze_backbone=False,
        )
        state = torch.load(weights_path, map_location=device)
        model.load_state_dict(state, strict=False)
        model.to(device).eval()
        return model, model_name, info
    except Exception as e:
        print(f"[WARN] Could not load model: {e}. Running in demo mode.")
        return None, model_name, info


# ─── Demo (no real model) ──────────────────────────────────────────────────────

def _demo_predict(tensor: torch.Tensor) -> float:
    """
    Deterministic pseudo-inference used when no trained model is available.
    Derives a consistent score from pixel statistics.
    """
    arr = tensor.numpy()
    seed = int(abs(arr.mean()) * 1e5) % (2 ** 31)
    rng = np.random.default_rng(seed)
    score = float(np.clip(abs(arr.mean()) * 3 + rng.normal(0.40, 0.12), 0.05, 0.95))
    return score


# ─── Main inference class ──────────────────────────────────────────────────────

class ParkinsonsInference:
    """
    High-level prediction interface.

    Instantiate once (cached), then call predict_*() methods.
    """

    def __init__(self, save_dir: str = SAVE_DIR, device: str = DEVICE,
                 threshold: Optional[float] = None):
        self.device = device
        self.threshold = threshold
        self.model, self.model_name, self.model_info = load_best_model(save_dir, device)
        self.demo_mode = self.model is None

        mode = "DEMO (no trained weights)" if self.demo_mode else "LIVE"
        print(f"[Inference] Mode={mode}  Model={self.model_name}  Device={device}")

    def _run_model(self, tensor: torch.Tensor) -> float:
        """Run forward pass, return probability [0,1]."""
        if self.demo_mode:
            return _demo_predict(tensor)
        with torch.no_grad():
            logit = self.model(tensor.to(self.device))
            prob  = torch.sigmoid(logit).item()
        return float(prob)

    # ── Image prediction ──

    def predict_image(
        self,
        file_obj,
        filename: str = "",
        size: int = 224,
    ) -> Dict[str, Any]:
        """
        Predict from an image file-like object.

        Returns dict with keys:
            probability, confidence, model_name, error (None or str)
        """
        try:
            validate_image_file(file_obj, filename)
            tensor = preprocess_image_to_tensor(file_obj, size=size)
            prob   = self._run_model(tensor)
        except ValidationError as e:
            return self._error_result(str(e))
        except Exception as e:
            return self._error_result(f"File could not be processed. ({type(e).__name__})")

        return self._format_result(prob)

    # ── Video prediction ──

    def predict_video(
        self,
        file_obj,
        filename: str = ".mp4",
        num_frames: int = 30,
        img_size: int = 112,
    ) -> Dict[str, Any]:
        """
        Predict from a video file-like object.

        For multi-modal systems the video model is separate; here we use
        a frame-level mean prediction via the image backbone as fallback.
        """
        tmp_path = None
        try:
            validate_video_file(file_obj, filename)
            tensor, tmp_path = preprocess_video_file(
                file_obj, filename, num_frames=num_frames, img_size=img_size
            )
            # tensor shape: [1, T, 3, H, W]
            # Use mean of per-frame predictions (simple but effective for demo)
            B, T, C, H, W = tensor.shape
            frame_probs = []
            for t in range(T):
                frame_tensor = tensor[0, t].unsqueeze(0)  # [1, 3, H, W]
                # Resize to match image model's expected input
                import torch.nn.functional as F
                frame_resized = F.interpolate(frame_tensor, size=(224, 224), mode="bilinear",
                                               align_corners=False)
                frame_probs.append(self._run_model(frame_resized))
            prob = float(np.mean(frame_probs))
        except ValidationError as e:
            return self._error_result(str(e))
        except Exception as e:
            return self._error_result(f"File could not be processed. ({type(e).__name__})")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

        return self._format_result(prob)

    # ── Helpers ──

    def _format_result(self, prob: float) -> Dict[str, Any]:
        confidence = max(prob, 1.0 - prob)
        return {
            "probability": round(prob, 4),
            "confidence": round(confidence, 4),
            "confidence_pct": round(prob * 100, 1),
            "model_name":  self.model_name,
            "demo_mode":   self.demo_mode,
            "error":       None,
        }

    @staticmethod
    def _error_result(message: str) -> Dict[str, Any]:
        return {
            "probability":     None,
            "confidence":      None,
            "confidence_pct":  None,
            "model_name":      None,
            "demo_mode":       True,
            "error":           message,
        }

    def get_model_metrics(self) -> Dict:
        """Return stored test metrics for the best model."""
        return {
            "accuracy":  self.model_info.get("accuracy"),
            "precision": self.model_info.get("precision"),
            "recall":    self.model_info.get("recall"),
            "f1":        self.model_info.get("f1"),
            "all_results": self.model_info.get("all_results", {}),
        }
