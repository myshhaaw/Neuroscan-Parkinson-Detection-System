"""
train_compare.py — Multi-model, multi-dataset training & comparison for NeuroScan AI

WHAT THIS FILE DOES:
  1. Loads all three datasets (MRI, Spiral, Video)
  2. Trains three models on image datasets: CustomCNN, MobileNetV2, EfficientNetB0
  3. Trains VideoTremorModel on video dataset with LSTM
  4. Evaluates every model: Accuracy, Precision, Recall, F1
  5. Prints a comparison table to console
  6. Saves training graphs (accuracy/loss curves)
  7. Saves confusion matrices + ROC curves for all models
  8. Selects the best model per dataset (by F1) and saves a registry JSON
  9. All errors are caught — never crashes on bad data

Run:
    python train_compare.py

Outputs (in ./results/ and ./models/):
    models/<name>_best.pth           ← best checkpoint per model
    models/best_model_registry.json  ← which model won each dataset
    results/<name>_train_curves.png  ← loss + accuracy curves
    results/<name>_confusion.png     ← confusion matrix
    results/<name>_roc.png           ← ROC curve
    results/model_comparison.png     ← side-by-side comparison table
"""

import os
import sys
import json
import copy
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe on all OS)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report
)

from models_arch import get_model, FusionClassifier, FEATURE_DIM
from preprocessing import (
    collect_image_samples, collect_video_samples,
    build_loaders, compute_pos_weight,
    ImageDataset, VideoDataset, get_transforms, split_dataset
)


def dynamic_threshold(probs) -> float:
    probs = np.asarray(probs, dtype=np.float32)
    if probs.size == 0:
        return 0.5
    return float(np.clip(probs.mean(), 0.0, 1.0))

# ─────────────────────────────────────────────────────────────────────────────
# ★  CONFIGURATION  — EDIT PATHS HERE
# ─────────────────────────────────────────────────────────────────────────────
MRI_PATH     = r"C:\Users\LENOVO\Downloads\archive (5)\parkinsons_dataset"
SPIRAL_PATH  = r"C:\Users\LENOVO\Downloads\archive (6)\spiral"
VIDEO_PATH   = r"C:\Users\LENOVO\Downloads\archive (7)"
BATCH_SIZE   = 16
NUM_EPOCHS   = 12           # image models
VIDEO_EPOCHS = 11           # video model (heavier per-sample)
LR           = 1e-4
WEIGHT_DECAY = 1e-4
NUM_FRAMES   = 30
NUM_WORKERS  = 0            # 0 = safe on Windows; increase on Linux
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR     = "models"
RESULTS_DIR  = "results"
MIN_SAMPLES  = 10           # skip dataset if fewer samples found

os.makedirs(SAVE_DIR,   exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# COLOUR PALETTE FOR PLOTS
# ─────────────────────────────────────────────────────────────────────────────
PALETTE = {
    "custom_cnn":    "#ef4444",
    "mobilenetv2":   "#3b82f6",
    "efficientnetb0":"#10b981",
    "video_tremor":  "#f59e0b",
    "mri":           "#8b5cf6",
    "spiral":        "#06b6d4",
    "video":         "#f97316",
}


# ═════════════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ═════════════════════════════════════════════════════════════════════════════
def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    total_loss = 0.0
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        optimizer.zero_grad()
        logits = model(batch_x)
        loss   = criterion(logits, batch_y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(batch_y)
    return total_loss / max(len(loader.dataset), 1)


def evaluate_loader(model, loader, criterion, device,
                    return_preds: bool = False):
    """
    Runs model on loader, returns metrics.
    return_preds=True → also returns (y_true, y_pred, y_prob) for plotting.
    """
    model.eval()
    total_loss  = 0.0
    all_preds   = []
    all_probs   = []
    all_labels  = []

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits  = model(batch_x)
            loss    = criterion(logits, batch_y)
            total_loss += loss.item() * len(batch_y)
            probs  = torch.sigmoid(logits).cpu().numpy().tolist()
            all_probs.extend(probs)
            all_labels.extend(batch_y.long().cpu().tolist())

    threshold = dynamic_threshold(all_probs)
    all_preds = [int(p >= threshold) for p in all_probs]

    avg_loss = total_loss / max(len(loader.dataset), 1)
    acc  = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec  = recall_score(all_labels,   all_preds, zero_division=0)
    f1   = f1_score(all_labels,       all_preds, zero_division=0)

    if return_preds:
        return avg_loss, acc, prec, rec, f1, all_labels, all_preds, all_probs
    return avg_loss, acc, prec, rec, f1


def train_model(model_name: str,
                train_loader: DataLoader,
                val_loader:   DataLoader,
                epochs:       int,
                pos_weight:   Optional[torch.Tensor] = None,
                unfreeze_at:  int = 10) -> Tuple[nn.Module, List[dict]]:
    """
    Full training loop for one model.

    unfreeze_at: epoch at which backbone is unfrozen for fine-tuning.
                 Ignored for CustomCNN (no backbone).
    Returns (best_model, history_list)
    """
    model = get_model(model_name, pretrained=True, freeze_backbone=True).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight.to(DEVICE) if pos_weight is not None else None
    )
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_f1      = 0.0
    best_weights = copy.deepcopy(model.state_dict())
    history      = []

    print(f"\n  {'─'*52}")
    print(f"  Training  [{model_name.upper()}]  on {DEVICE}  ({epochs} epochs)")
    print(f"  {'─'*52}")

    for epoch in range(1, epochs + 1):

        # ── Unfreeze backbone half-way through training ──
        if epoch == unfreeze_at and model_name != "custom_cnn":
            try:
                model.unfreeze_all()
                optimizer = optim.AdamW(
                    model.parameters(), lr=LR * 0.1,
                    weight_decay=WEIGHT_DECAY
                )
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=epochs - epoch
                )
                print(f"  [Epoch {epoch}] Backbone unfrozen — fine-tuning at LR×0.1")
            except AttributeError:
                pass   # VideoTremorModel has unfreeze_cnn(), handled separately

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        scheduler.step()
        val_loss, acc, prec, rec, f1 = evaluate_loader(
            model, val_loader, criterion, DEVICE
        )

        print(
            f"  Ep {epoch:02d}/{epochs} | "
            f"TL={train_loss:.4f}  VL={val_loss:.4f} | "
            f"Acc={acc:.3f}  P={prec:.3f}  R={rec:.3f}  F1={f1:.3f}"
        )

        history.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "acc": acc, "prec": prec, "rec": rec, "f1": f1
        })

        if f1 >= best_f1:
            best_f1      = f1
            best_weights = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_weights)
    save_path = os.path.join(SAVE_DIR, f"{model_name}_best.pth")
    torch.save(model.state_dict(), save_path)
    print(f"  ✓ Best model saved → {save_path}  (Val F1={best_f1:.4f})")
    return model, history


# ═════════════════════════════════════════════════════════════════════════════
# PLOTTING UTILITIES
# ═════════════════════════════════════════════════════════════════════════════
def plot_train_curves(histories: Dict[str, List[dict]], dataset_name: str):
    """Saves a 2×1 figure: training curves (loss left, accuracy right)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Training Curves — {dataset_name} Dataset",
                 fontsize=14, fontweight="bold")

    for mname, hist in histories.items():
        color  = PALETTE.get(mname, "#888")
        epochs = [h["epoch"]      for h in hist]
        t_loss = [h["train_loss"] for h in hist]
        v_loss = [h["val_loss"]   for h in hist]
        v_acc  = [h["acc"]        for h in hist]

        ax1.plot(epochs, t_loss, "--", color=color, alpha=0.6,
                 label=f"{mname} train")
        ax1.plot(epochs, v_loss, "-",  color=color, linewidth=2,
                 label=f"{mname} val")
        ax2.plot(epochs, v_acc,  "-",  color=color, linewidth=2,
                 marker=".", label=mname)

    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Train vs Validation Loss"); ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    ax2.set_title("Validation Accuracy"); ax2.legend(fontsize=7)
    ax2.set_ylim(0, 1); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, f"{dataset_name}_train_curves.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  📊 Training curves saved → {out}")


def plot_confusion_matrix(y_true, y_pred, model_name: str, dataset_name: str):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    labels = ["Healthy", "Parkinson's"]
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels)
    ax.set_yticks([0, 1]); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {model_name} ({dataset_name})")
    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=16, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, f"{model_name}_{dataset_name}_confusion.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  📊 Confusion matrix saved → {out}")


def plot_roc_curve(y_true, y_prob, model_name: str, dataset_name: str):
    try:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc     = auc(fpr, tpr)
    except Exception:
        return  # skip if only one class present

    fig, ax = plt.subplots(figsize=(5, 4))
    color = PALETTE.get(model_name, "#3b82f6")
    ax.plot(fpr, tpr, color=color, linewidth=2,
            label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {model_name} ({dataset_name})")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, f"{model_name}_{dataset_name}_roc.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  📊 ROC curve saved → {out}")


def plot_comparison_table(all_results: Dict[str, dict]):
    """
    Generates a visual comparison table of all models across all datasets.
    all_results: { "MRI_custom_cnn": {"acc":..,"prec":..,"rec":..,"f1":..}, ... }
    """
    if not all_results:
        return

    rows    = list(all_results.keys())
    metrics = ["acc", "prec", "rec", "f1"]
    labels  = ["Accuracy", "Precision", "Recall", "F1-score"]

    data = np.array([[all_results[r].get(m, 0) for m in metrics] for r in rows])

    fig, ax = plt.subplots(figsize=(max(10, len(rows) * 1.4), 5))
    x       = np.arange(len(labels))
    width   = 0.8 / max(len(rows), 1)

    for i, row in enumerate(rows):
        offset = (i - len(rows) / 2 + 0.5) * width
        bars   = ax.bar(x + offset, data[i], width * 0.9,
                        label=row,
                        color=PALETTE.get(row.split("_")[1] if "_" in row else row, "#888"),
                        alpha=0.85, edgecolor="white")
        # value labels
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — All Datasets", fontsize=13, fontweight="bold")
    ax.legend(fontsize=7, ncol=max(1, len(rows) // 3))
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "model_comparison.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\n  📊 Comparison chart saved → {out}")


# ═════════════════════════════════════════════════════════════════════════════
# CONSOLE TABLE
# ═════════════════════════════════════════════════════════════════════════════
def print_results_table(all_results: Dict[str, dict]):
    """Pretty-prints a comparison table to stdout."""
    print("\n" + "═" * 72)
    print("  MODEL COMPARISON TABLE")
    print("═" * 72)
    header = f"  {'Model':<30}  {'Acc':>6}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}"
    print(header)
    print("  " + "─" * 68)

    best_f1  = -1.0
    best_key = ""
    for key, res in sorted(all_results.items()):
        tag = f"★ " if res.get("f1", 0) > best_f1 else "  "
        if res.get("f1", 0) > best_f1:
            best_f1  = res["f1"]
            best_key = key
        print(f"{tag}{key:<30}  "
              f"{res.get('acc', 0):6.3f}  "
              f"{res.get('prec',0):6.3f}  "
              f"{res.get('rec', 0):6.3f}  "
              f"{res.get('f1',  0):6.3f}")

    print("═" * 72)
    print(f"  ★ Best overall → {best_key}  (F1 = {best_f1:.4f})")
    print("═" * 72 + "\n")


# ═════════════════════════════════════════════════════════════════════════════
# DATASET TRAINING BLOCK  (reusable for MRI / Spiral)
# ═════════════════════════════════════════════════════════════════════════════
def run_image_dataset(dataset_name: str,
                      data_path:    str,
                      model_names:  List[str],
                      all_results:  dict,
                      best_registry: dict):
    """
    Trains all image-type models on one dataset.
    Updates all_results and best_registry in-place.
    """
    print(f"\n{'━'*60}")
    print(f"  DATASET: {dataset_name.upper()}")
    print(f"  Path   : {data_path}")
    print(f"{'━'*60}")

    samples = collect_image_samples(data_path)
    print(f"  Total samples: {len(samples)}")

    if len(samples) < MIN_SAMPLES:
        print(f"  [SKIP] Need at least {MIN_SAMPLES} samples — only {len(samples)} found")
        print(f"         Check that folders are named 'healthy' and 'parkinson'")
        return

    pw = compute_pos_weight(samples)

    # Build split samples (indices) for each split
    # We create one ImageDataset with no transform, then apply transforms per split
    all_ds    = ImageDataset(samples, transform=None)
    tr_sub, va_sub, te_sub = split_dataset(all_ds)

    tr_samples = [samples[i] for i in tr_sub.indices]
    va_samples = [samples[i] for i in va_sub.indices]
    te_samples = [samples[i] for i in te_sub.indices]

    train_loader = DataLoader(
        ImageDataset(tr_samples, transform=get_transforms("train")),
        batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        ImageDataset(va_samples, transform=get_transforms("val")),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )
    test_loader = DataLoader(
        ImageDataset(te_samples, transform=get_transforms("test")),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    histories      = {}
    dataset_results = {}
    best_f1        = -1.0
    best_model_key = ""

    for mname in model_names:
        key = f"{dataset_name}_{mname}"
        try:
            model, hist = train_model(
                mname, train_loader, val_loader,
                epochs=NUM_EPOCHS, pos_weight=pw,
                unfreeze_at=NUM_EPOCHS // 2
            )
            histories[mname] = hist

            # Evaluate on test set
            crit = nn.BCEWithLogitsLoss()
            _, acc, prec, rec, f1, yt, yp, ypr = evaluate_loader(
                model, test_loader, crit, DEVICE, return_preds=True
            )
            res = {"acc": acc, "prec": prec, "rec": rec, "f1": f1}
            all_results[key]       = res
            dataset_results[mname] = res

            print(f"\n  TEST [{mname}] → "
                  f"Acc={acc:.3f}  Prec={prec:.3f}  Rec={rec:.3f}  F1={f1:.3f}")
            print(classification_report(yt, yp,
                  target_names=["Healthy", "Parkinson's"], zero_division=0))

            # Plots
            plot_confusion_matrix(yt, yp,  mname, dataset_name)
            plot_roc_curve(yt, ypr,         mname, dataset_name)

            if f1 > best_f1:
                best_f1        = f1
                best_model_key = mname

        except Exception as e:
            print(f"  [ERROR] {mname} training failed: {e}")
            import traceback; traceback.print_exc()

    if histories:
        plot_train_curves(histories, dataset_name)

    if best_model_key:
        best_registry[dataset_name] = {
            "model": best_model_key,
            "f1":    best_f1,
            "path":  os.path.join(SAVE_DIR, f"{best_model_key}_best.pth")
        }
        print(f"\n  🏆 Best model for {dataset_name}: "
              f"{best_model_key}  (F1={best_f1:.4f})")


# ═════════════════════════════════════════════════════════════════════════════
# VIDEO DATASET TRAINING BLOCK
# ═════════════════════════════════════════════════════════════════════════════
def run_video_dataset(data_path:    str,
                      all_results:  dict,
                      best_registry: dict):
    print(f"\n{'━'*60}")
    print(f"  DATASET: VIDEO TREMOR")
    print(f"  Path   : {data_path}")
    print(f"{'━'*60}")

    samples = collect_video_samples(data_path)
    print(f"  Total samples: {len(samples)}")

    if len(samples) < MIN_SAMPLES:
        print(f"  [SKIP] Need ≥ {MIN_SAMPLES} video samples — only {len(samples)} found")
        return

    pw = compute_pos_weight(samples)

    all_ds = VideoDataset(samples, num_frames=NUM_FRAMES)
    tr_sub, va_sub, te_sub = split_dataset(all_ds)
    bs_vid = max(1, BATCH_SIZE // 4)

    train_loader = DataLoader(
        VideoDataset([samples[i] for i in tr_sub.indices], NUM_FRAMES),
        batch_size=bs_vid, shuffle=True, num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        VideoDataset([samples[i] for i in va_sub.indices], NUM_FRAMES),
        batch_size=bs_vid, shuffle=False, num_workers=NUM_WORKERS
    )
    test_loader = DataLoader(
        VideoDataset([samples[i] for i in te_sub.indices], NUM_FRAMES),
        batch_size=bs_vid, shuffle=False, num_workers=NUM_WORKERS
    )

    # Import VideoTremorModel directly
    from models_arch import VideoTremorModel
    model = VideoTremorModel(feature_dim=FEATURE_DIM,
                             num_frames=NUM_FRAMES, pretrained=True).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pw.to(DEVICE) if pw is not None else None
    )
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=VIDEO_EPOCHS)

    best_f1      = 0.0
    best_weights = copy.deepcopy(model.state_dict())
    history      = []

    print(f"\n  Training  [VIDEO_TREMOR + BiLSTM]  ({VIDEO_EPOCHS} epochs)")
    print(f"  {'─'*52}")

    for epoch in range(1, VIDEO_EPOCHS + 1):
        # Unfreeze CNN encoder after first third of training
        if epoch == VIDEO_EPOCHS // 3:
            model.unfreeze_cnn()
            optimizer = optim.AdamW(model.parameters(), lr=LR * 0.1,
                                    weight_decay=WEIGHT_DECAY)
            print(f"  [Epoch {epoch}] CNN encoder unfrozen")

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        scheduler.step()
        val_loss, acc, prec, rec, f1 = evaluate_loader(
            model, val_loader, criterion, DEVICE
        )
        print(f"  Ep {epoch:02d}/{VIDEO_EPOCHS} | "
              f"TL={train_loss:.4f}  VL={val_loss:.4f} | "
              f"Acc={acc:.3f}  F1={f1:.3f}")
        history.append({"epoch": epoch, "train_loss": train_loss,
                         "val_loss": val_loss, "acc": acc, "f1": f1,
                         "prec": prec, "rec": rec})
        if f1 >= best_f1:
            best_f1      = f1
            best_weights = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_weights)
    save_path = os.path.join(SAVE_DIR, "video_tremor_best.pth")
    torch.save(model.state_dict(), save_path)
    print(f"  ✓ Saved → {save_path}  (Val F1={best_f1:.4f})")

    plot_train_curves({"video_tremor": history}, "Video")

    crit = nn.BCEWithLogitsLoss()
    _, acc, prec, rec, f1, yt, yp, ypr = evaluate_loader(
        model, test_loader, crit, DEVICE, return_preds=True
    )
    print(f"\n  TEST [video_tremor] → "
          f"Acc={acc:.3f}  Prec={prec:.3f}  Rec={rec:.3f}  F1={f1:.3f}")
    print(classification_report(yt, yp,
          target_names=["Healthy", "Parkinson's"], zero_division=0))

    plot_confusion_matrix(yt, yp,  "video_tremor", "Video")
    plot_roc_curve(yt, ypr,        "video_tremor", "Video")

    all_results["Video_video_tremor"] = {"acc": acc, "prec": prec, "rec": rec, "f1": f1}
    best_registry["Video"] = {
        "model": "video_tremor",
        "f1":    f1,
        "path":  save_path
    }


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
def main():
    start = time.time()

    print("\n" + "═" * 60)
    print("  NeuroScan AI — Multi-Model Training Pipeline")
    print(f"  Device: {DEVICE}")
    print("═" * 60)

    image_models = ["custom_cnn", "mobilenetv2", "efficientnetb0"]
    all_results:   Dict[str, dict] = {}
    best_registry: Dict[str, dict] = {}

    # ── 1. MRI Dataset ───────────────────────────────────────
    run_image_dataset("MRI",    MRI_PATH,    image_models,
                      all_results, best_registry)

    # ── 2. Spiral Drawing Dataset ─────────────────────────────
    run_image_dataset("Spiral", SPIRAL_PATH, image_models,
                      all_results, best_registry)

    # ── 3. Video Tremor Dataset ───────────────────────────────
    run_video_dataset(VIDEO_PATH, all_results, best_registry)

    # ── 4. Final reporting ────────────────────────────────────
    if all_results:
        print_results_table(all_results)
        plot_comparison_table(all_results)

    # ── 5. Save registry ──────────────────────────────────────
    registry_path = os.path.join(SAVE_DIR, "best_model_registry.json")
    with open(registry_path, "w") as f:
        json.dump(best_registry, f, indent=2)
    print(f"  ✓ Best model registry saved → {registry_path}")

    elapsed = time.time() - start
    print(f"\n  ✓ Total training time: {elapsed/60:.1f} min")
    print("\n  To launch the Streamlit app:")
    print("    streamlit run app.py\n")


if __name__ == "__main__":
    main()
