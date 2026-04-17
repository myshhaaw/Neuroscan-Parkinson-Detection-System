"""
evaluate.py — NeuroScan AI v3.0  Evaluation & Visualisation
────────────────────────────────────────────────────────────────────────────────
Generates (all saved to results/):
  • confusion_matrix.png     — grid of per-model confusion matrices
  • accuracy_graph.png       — train vs val accuracy per model
  • loss_graph.png           — train vs val loss per model
  • roc_curve.png            — multi-model ROC curves on one plot
  • metrics_dashboard.png    — grouped bar chart (Acc/Prec/Rec/F1/Spec)
  • comparison_table.png     — rendered HTML-style table
  • metrics_report.json      — all scalar metrics

Usage:
    from evaluate import run_full_evaluation
    run_full_evaluation(histories_dict, test_results_dict)

Or standalone demo:
    python evaluate.py
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    roc_curve, auc,
)

# ── Output directory ──────────────────────────────────────────────────────────
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Colour palette ────────────────────────────────────────────────────────────
BG      = "#0a0e1a"
SURFACE = "#111827"
CARD    = "#161f30"
ACCENT  = "#00e5ff"
PURPLE  = "#7c3aed"
GREEN   = "#10b981"
YELLOW  = "#f59e0b"
RED     = "#ef4444"
ORANGE  = "#f97316"
PINK    = "#ec4899"
TEXT    = "#e2e8f0"
MUTED   = "#64748b"
BORDER  = "#1e293b"

# Palette for however many models we get
_PALETTE = [ACCENT, PURPLE, GREEN, YELLOW, RED, ORANGE, PINK,
            "#60a5fa", "#a78bfa", "#34d399"]

MODEL_LABELS_DEFAULT = {
    "mri":              "MRI",
    "mri_custom":       "MRI · CustomCNN",
    "mri_mobilenet":    "MRI · MobileNetV2",
    "mri_efficientnet": "MRI · EfficientNetB0",
    "video":            "Video Tremor",
    "scribble":         "Scribble",
    "scribble_custom":       "Scribble · CustomCNN",
    "scribble_mobilenet":    "Scribble · MobileNetV2",
    "scribble_efficientnet": "Scribble · EfficientNetB0",
    "fusion":           "Multi-Modal Fusion",
}


def dynamic_threshold(probs) -> float:
    probs = np.asarray(probs, dtype=np.float32)
    if probs.size == 0:
        return 0.5
    return float(np.clip(probs.mean(), 0.0, 1.0))


def _label(name: str) -> str:
    return MODEL_LABELS_DEFAULT.get(name, name.replace("_", " ").title())


def _color(i: int) -> str:
    return _PALETTE[i % len(_PALETTE)]


def _theme():
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor":  SURFACE,
        "axes.edgecolor":  BORDER, "axes.labelcolor": TEXT,
        "axes.titlecolor": TEXT,   "xtick.color":     MUTED,
        "ytick.color":     MUTED,  "grid.color":      BORDER,
        "grid.linewidth":  0.6,    "text.color":      TEXT,
        "legend.facecolor": CARD,  "legend.edgecolor": BORDER,
        "font.family":     "monospace",
        "font.size": 9, "axes.titlesize": 11, "axes.labelsize": 9,
    })


# ══════════════════════════════════════════════════════════════════════════════
# 1.  SCALAR METRICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(y_true, y_pred, y_prob=None,
                    model_name="model") -> Dict:
    y_true = list(y_true)
    if y_prob is not None:
        threshold = dynamic_threshold(y_prob)
        y_pred = [int(float(p) >= threshold) for p in y_prob]
    else:
        y_pred = list(y_pred)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, int(cm[0, 0]))

    m = {
        "model":      model_name,
        "timestamp":  datetime.now().isoformat(timespec="seconds"),
        "n_samples":  len(y_true),
        "n_positive": int(sum(y_true)),
        "n_negative": int(len(y_true) - sum(y_true)),
        "accuracy":   round(float(accuracy_score(y_true, y_pred)), 4),
        "precision":  round(float(precision_score(y_true, y_pred,  zero_division=0)), 4),
        "recall":     round(float(recall_score(y_true, y_pred,     zero_division=0)), 4),
        "f1_score":   round(float(f1_score(y_true, y_pred,         zero_division=0)), 4),
        "specificity":round(float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0, 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp),
                             "fn": int(fn), "tp": int(tp)},
        "classification_report": classification_report(
            y_true, y_pred,
            target_names=["No PD", "Parkinson's"],
            zero_division=0, output_dict=True,
        ),
    }
    if y_prob is not None:
        try:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            m["roc_auc"]  = round(float(auc(fpr, tpr)), 4)
            m["_roc_fpr"] = [round(v, 4) for v in fpr.tolist()]
            m["_roc_tpr"] = [round(v, 4) for v in tpr.tolist()]
        except Exception:
            pass
    return m


# ══════════════════════════════════════════════════════════════════════════════
# 2.  CONFUSION MATRIX
# ══════════════════════════════════════════════════════════════════════════════

def plot_confusion_matrix(test_results: Dict, save_path: str = None) -> str:
    _theme()
    save_path = save_path or os.path.join(RESULTS_DIR, "confusion_matrix.png")
    models = list(test_results.keys())
    n = len(models)
    cols = min(n, 4)
    rows = max(1, (n + cols - 1) // cols)

    fig, axes = plt.subplots(rows, cols,
                             figsize=(4.5 * cols, 4.5 * rows),
                             facecolor=BG)
    fig.suptitle("Confusion Matrices — NeuroScan AI",
                 fontsize=14, color=TEXT, fontweight="bold", y=1.01)

    from matplotlib.colors import LinearSegmentedColormap
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    for i, name in enumerate(models):
        ax    = axes_flat[i]
        data  = test_results[name]
        cm    = confusion_matrix(data["y_true"], data["y_pred"])
        color = _color(i)
        cmap  = LinearSegmentedColormap.from_list("nc", [SURFACE, color], N=256)

        sns.heatmap(
            cm, annot=True, fmt="d", ax=ax, cmap=cmap,
            linewidths=1, linecolor=BORDER,
            annot_kws={"size": 16, "weight": "bold", "color": BG},
            cbar=False,
            xticklabels=["No PD", "PD"],
            yticklabels=["No PD", "PD"],
        )
        acc = accuracy_score(data["y_true"], data["y_pred"])
        f1  = f1_score(data["y_true"], data["y_pred"], zero_division=0)
        ax.set_facecolor(SURFACE)
        ax.set_title(_label(name), color=color, fontsize=10,
                     fontweight="bold", pad=8)
        ax.set_xlabel(f"Predicted | Acc={acc:.2f}  F1={f1:.2f}",
                      color=MUTED, fontsize=8)
        ax.set_ylabel("Actual", color=MUTED, fontsize=8)
        ax.tick_params(colors=MUTED)

    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout(pad=2.0)
    fig.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close(fig)
    print(f"[Eval] ✓ confusion_matrix → {save_path}")
    return save_path


# ══════════════════════════════════════════════════════════════════════════════
# 3.  ACCURACY GRAPH
# ══════════════════════════════════════════════════════════════════════════════

def plot_accuracy_graph(histories: Dict, save_path: str = None) -> str:
    _theme()
    save_path = save_path or os.path.join(RESULTS_DIR, "accuracy_graph.png")
    models = [m for m, h in histories.items() if h]
    if not models:
        return ""
    n    = len(models)
    cols = min(n, 3)
    rows = max(1, (n + cols - 1) // cols)

    fig, axes = plt.subplots(rows, cols,
                             figsize=(6 * cols, 4 * rows),
                             facecolor=BG)
    fig.suptitle("Training vs Validation Accuracy",
                 fontsize=14, color=TEXT, fontweight="bold", y=1.01)
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    for i, name in enumerate(models):
        ax    = axes_flat[i]
        hist  = histories[name]
        eps   = [h["epoch"] for h in hist]
        t_acc = [h.get("acc", 0) for h in hist]
        v_acc = [h.get("val_acc", h.get("acc", 0)) for h in hist]
        color = _color(i)

        ax.set_facecolor(SURFACE)
        ax.grid(True, alpha=0.3)
        ax.fill_between(eps, t_acc, alpha=0.1, color=color)
        ax.plot(eps, t_acc, color=color, lw=2, label="Train", marker="o",
                markersize=3, markevery=max(1, len(eps) // 8))
        ax.plot(eps, v_acc, color=MUTED, lw=1.8, ls="--", label="Val",
                marker="s", markersize=3, markevery=max(1, len(eps) // 8))
        best = int(np.argmax(v_acc))
        ax.scatter(eps[best], v_acc[best], color=color, s=70,
                   zorder=5, edgecolors=BG, lw=1.5)
        ax.annotate(f"Best {v_acc[best]:.3f}",
                    xy=(eps[best], v_acc[best]),
                    xytext=(6, 6), textcoords="offset points",
                    color=color, fontsize=8)
        ax.set_title(_label(name), color=color, fontsize=10, fontweight="bold")
        ax.set_xlabel("Epoch", color=MUTED)
        ax.set_ylabel("Accuracy", color=MUTED)
        ax.set_ylim(0, 1.08)
        ax.legend(loc="lower right", fontsize=8)
        ax.spines[:].set_color(BORDER)

    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout(pad=2.0)
    fig.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close(fig)
    print(f"[Eval] ✓ accuracy_graph → {save_path}")
    return save_path


# ══════════════════════════════════════════════════════════════════════════════
# 4.  LOSS GRAPH
# ══════════════════════════════════════════════════════════════════════════════

def plot_loss_graph(histories: Dict, save_path: str = None) -> str:
    _theme()
    save_path = save_path or os.path.join(RESULTS_DIR, "loss_graph.png")
    models = [m for m, h in histories.items() if h]
    if not models:
        return ""
    n    = len(models)
    cols = min(n, 3)
    rows = max(1, (n + cols - 1) // cols)

    fig, axes = plt.subplots(rows, cols,
                             figsize=(6 * cols, 4 * rows),
                             facecolor=BG)
    fig.suptitle("Training vs Validation Loss",
                 fontsize=14, color=TEXT, fontweight="bold", y=1.01)
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    for i, name in enumerate(models):
        ax   = axes_flat[i]
        hist = histories[name]
        eps  = [h["epoch"]      for h in hist]
        tl   = [h["train_loss"] for h in hist]
        vl   = [h["val_loss"]   for h in hist]
        color = _color(i)

        ax.set_facecolor(SURFACE)
        ax.grid(True, alpha=0.3)
        ax.fill_between(eps, tl, alpha=0.1, color=color)
        ax.plot(eps, tl, color=color, lw=2, label="Train", marker="o",
                markersize=3, markevery=max(1, len(eps) // 8))
        ax.plot(eps, vl, color=RED, lw=1.8, ls="--", label="Val",
                marker="s", markersize=3, markevery=max(1, len(eps) // 8))
        best = int(np.argmin(vl))
        ax.scatter(eps[best], vl[best], color=RED, s=70,
                   zorder=5, edgecolors=BG, lw=1.5)
        ax.annotate(f"Min {vl[best]:.3f}",
                    xy=(eps[best], vl[best]),
                    xytext=(6, 6), textcoords="offset points",
                    color=RED, fontsize=8)
        ax.set_title(_label(name), color=color, fontsize=10, fontweight="bold")
        ax.set_xlabel("Epoch", color=MUTED)
        ax.set_ylabel("Loss (BCE)", color=MUTED)
        ax.legend(loc="upper right", fontsize=8)
        ax.spines[:].set_color(BORDER)

    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout(pad=2.0)
    fig.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close(fig)
    print(f"[Eval] ✓ loss_graph → {save_path}")
    return save_path


# ══════════════════════════════════════════════════════════════════════════════
# 5.  ROC CURVES  (multi-model on one plot)
# ══════════════════════════════════════════════════════════════════════════════

def plot_roc_curves(all_metrics: Dict, save_path: str = None) -> str:
    _theme()
    save_path = save_path or os.path.join(RESULTS_DIR, "roc_curve.png")

    fig, ax = plt.subplots(figsize=(7, 6), facecolor=BG)
    ax.set_facecolor(SURFACE)
    ax.grid(True, alpha=0.3)

    # diagonal
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--", label="Random (AUC=0.50)")

    plotted = 0
    for i, (name, m) in enumerate(all_metrics.items()):
        fpr = m.get("_roc_fpr")
        tpr = m.get("_roc_tpr")
        auc_val = m.get("roc_auc")
        if fpr is None or tpr is None:
            continue
        color = _color(i)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{_label(name)}  (AUC={auc_val:.3f})")
        plotted += 1

    if plotted == 0:
        ax.text(0.5, 0.5, "No probability outputs available",
                ha="center", va="center", color=MUTED, fontsize=11,
                transform=ax.transAxes)

    ax.set_xlabel("False Positive Rate", color=MUTED)
    ax.set_ylabel("True Positive Rate", color=MUTED)
    ax.set_title("ROC Curves — All Models", color=TEXT,
                 fontsize=13, fontweight="bold")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.legend(loc="lower right", fontsize=8)
    ax.spines[:].set_color(BORDER)

    plt.tight_layout(pad=2.0)
    fig.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close(fig)
    print(f"[Eval] ✓ roc_curve → {save_path}")
    return save_path


# ══════════════════════════════════════════════════════════════════════════════
# 6.  METRICS DASHBOARD  (grouped bar)
# ══════════════════════════════════════════════════════════════════════════════

def plot_metrics_dashboard(all_metrics: Dict, save_path: str = None) -> str:
    _theme()
    save_path = save_path or os.path.join(RESULTS_DIR, "metrics_dashboard.png")

    keys   = ["accuracy", "precision", "recall", "f1_score", "specificity"]
    labels = ["Accuracy", "Precision", "Recall", "F1-Score", "Specificity"]
    models = list(all_metrics.keys())
    x      = np.arange(len(keys))
    width  = 0.8 / max(len(models), 1)

    fig, ax = plt.subplots(figsize=(12, 5.5), facecolor=BG)
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis="y", alpha=0.3)

    for j, name in enumerate(models):
        m     = all_metrics[name]
        vals  = [m.get(k, 0.0) for k in keys]
        color = _color(j)
        offset = (j - len(models) / 2 + 0.5) * width
        bars  = ax.bar(x + offset, vals, width * 0.88,
                       label=_label(name), color=color,
                       alpha=0.85, edgecolor=BG, lw=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.012,
                    f"{v:.2f}", ha="center", va="bottom",
                    color=color, fontsize=7, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, color=TEXT)
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("Score", color=MUTED)
    ax.set_title("Multi-Model Performance Comparison",
                 color=TEXT, fontsize=13, fontweight="bold", pad=14)
    ax.spines[:].set_color(BORDER)
    ax.legend(loc="upper right", fontsize=8)
    ax.axhline(1.0, color=BORDER, lw=0.7, ls="--")

    plt.tight_layout(pad=2.0)
    fig.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close(fig)
    print(f"[Eval] ✓ metrics_dashboard → {save_path}")
    return save_path


# ══════════════════════════════════════════════════════════════════════════════
# 7.  COMPARISON TABLE IMAGE
# ══════════════════════════════════════════════════════════════════════════════

def plot_comparison_table(all_metrics: Dict, save_path: str = None) -> str:
    """Renders a clean comparison table as a figure."""
    _theme()
    save_path = save_path or os.path.join(RESULTS_DIR, "comparison_table.png")

    cols_show = ["accuracy", "precision", "recall", "f1_score",
                 "specificity", "roc_auc"]
    col_labels = ["Accuracy", "Precision", "Recall", "F1", "Specificity", "AUC-ROC"]
    models = list(all_metrics.keys())

    rows = []
    for name in models:
        m    = all_metrics[name]
        row  = [_label(name)]
        row += [f"{m.get(c, '—'):.3f}" if isinstance(m.get(c), float)
                else "—" for c in cols_show]
        rows.append(row)

    n_rows = len(rows)
    n_cols = len(col_labels) + 1  # +1 for model name

    fig, ax = plt.subplots(figsize=(n_cols * 1.6, n_rows * 0.55 + 1.2),
                            facecolor=BG)
    ax.set_facecolor(BG)
    ax.axis("off")

    header = ["Model"] + col_labels
    all_rows = [header] + rows
    n_all = len(all_rows)

    col_w  = 1.0 / n_cols
    row_h  = 1.0 / n_all

    for r, row in enumerate(all_rows):
        for c, cell in enumerate(row):
            is_header = (r == 0)
            is_best_col = (c > 0)

            # background
            if is_header:
                bg = CARD
                fg = ACCENT
                fw = "bold"
            elif r % 2 == 0:
                bg = SURFACE
                fg = TEXT
                fw = "normal"
            else:
                bg = CARD
                fg = TEXT
                fw = "normal"

            rect = plt.Rectangle(
                (c * col_w, 1 - (r + 1) * row_h),
                col_w, row_h,
                transform=ax.transAxes,
                facecolor=bg, edgecolor=BORDER, lw=0.5,
                clip_on=False,
            )
            ax.add_patch(rect)
            ax.text(
                c * col_w + col_w / 2,
                1 - (r + 0.5) * row_h,
                cell,
                transform=ax.transAxes,
                ha="center", va="center",
                fontsize=8.5, color=fg, fontweight=fw,
                fontfamily="monospace",
            )

    ax.set_title("Model Comparison Summary",
                 color=TEXT, fontsize=12, fontweight="bold", pad=10)

    plt.tight_layout(pad=0.5)
    fig.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close(fig)
    print(f"[Eval] ✓ comparison_table → {save_path}")
    return save_path


# ══════════════════════════════════════════════════════════════════════════════
# 8.  GRAD-CAM HEATMAP OVERLAY  (call separately with a real image)
# ══════════════════════════════════════════════════════════════════════════════

def save_gradcam_overlay(
    original_img,          # PIL Image
    cam_array: np.ndarray, # (H, W) float [0, 1]
    save_path: str,
    title: str = "GradCAM",
) -> str:
    """
    Blends the GradCAM heatmap on the original image and saves it.
    Returns save_path.
    """
    import cv2 as _cv2
    _theme()

    img_np = np.array(original_img.resize((224, 224))).astype(np.uint8)
    cam_resized = _cv2.resize(cam_array, (224, 224))
    heatmap = _cv2.applyColorMap(
        (cam_resized * 255).astype(np.uint8), _cv2.COLORMAP_JET
    )
    heatmap = _cv2.cvtColor(heatmap, _cv2.COLOR_BGR2RGB)
    overlay = (0.55 * img_np + 0.45 * heatmap).clip(0, 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(9, 3.5), facecolor=BG)
    for ax, data, ttl, cmap in zip(
        axes,
        [img_np, heatmap, overlay],
        ["Original", "GradCAM", "Overlay"],
        [None, None, None],
    ):
        ax.imshow(data, cmap=cmap)
        ax.set_title(ttl, color=ACCENT, fontsize=9)
        ax.axis("off")
        ax.set_facecolor(BG)

    fig.suptitle(title, color=TEXT, fontsize=11, fontweight="bold")
    plt.tight_layout(pad=1.0)
    fig.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close(fig)
    return save_path


# ══════════════════════════════════════════════════════════════════════════════
# MASTER ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def run_full_evaluation(
    histories: Dict[str, List[Dict]],
    test_results: Dict[str, Dict],
) -> Dict:
    """
    Compute all metrics and generate all visualisation artefacts.

    histories     : { model_name: [epoch_dict, ...] }
    test_results  : { model_name: {"y_true": [...], "y_pred": [...],
                                   "y_prob": [...]} }

    Returns master_report dict (also saved to results/metrics_report.json).
    """
    print("\n" + "─" * 55)
    print("  Evaluation & Visualisation")
    print("─" * 55)

    # Scalar metrics
    all_metrics = {}
    for name, data in test_results.items():
        m = compute_metrics(
            data["y_true"], data["y_pred"],
            data.get("y_prob"), name,
        )
        all_metrics[name] = m
        print(f"  [{_label(name):30s}] "
              f"Acc={m['accuracy']:.3f}  Prec={m['precision']:.3f}  "
              f"Rec={m['recall']:.3f}  F1={m['f1_score']:.3f}  "
              f"Spec={m['specificity']:.3f}  "
              f"AUC={m.get('roc_auc', '—')}")

    # Identify best overall model
    best_name = max(all_metrics, key=lambda k: all_metrics[k]["f1_score"])
    print(f"\n  ★ Best model: {_label(best_name)}  "
          f"(F1={all_metrics[best_name]['f1_score']:.3f})")

    # Plots
    cm_path   = plot_confusion_matrix(test_results)
    acc_path  = plot_accuracy_graph(histories) if any(histories.values()) else None
    loss_path = plot_loss_graph(histories)     if any(histories.values()) else None
    roc_path  = plot_roc_curves(all_metrics)
    dash_path = plot_metrics_dashboard(all_metrics)
    tbl_path  = plot_comparison_table(all_metrics)

    # JSON report
    # Strip private "_roc_*" arrays to keep JSON lean
    clean_metrics = {
        k: {mk: mv for mk, mv in m.items() if not mk.startswith("_")}
        for k, m in all_metrics.items()
    }
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "best_model":   best_name,
        "models":       clean_metrics,
        "image_paths": {
            "confusion_matrix":  cm_path,
            "accuracy_graph":    acc_path,
            "loss_graph":        loss_path,
            "roc_curve":         roc_path,
            "metrics_dashboard": dash_path,
            "comparison_table":  tbl_path,
        },
    }
    report_path = os.path.join(RESULTS_DIR, "metrics_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n  ✓ JSON report → {report_path}")
    print("─" * 55 + "\n")
    return report


# ══════════════════════════════════════════════════════════════════════════════
# STANDALONE DEMO  (synthetic data)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    rng = np.random.default_rng(42)

    def _hist(n=12, noisy=0.05):
        h = []
        for e in range(1, n + 1):
            prog = e / n
            h.append({
                "epoch":      e,
                "train_loss": max(0.05, 0.65 * (1 - prog) + rng.normal(0, noisy)),
                "val_loss":   max(0.08, 0.60 * (1 - prog * 0.85) + rng.normal(0, noisy * 1.5)),
                "acc":        min(0.98, 0.50 + 0.44 * prog + rng.normal(0, noisy)),
                "val_acc":    min(0.96, 0.50 + 0.40 * prog + rng.normal(0, noisy * 1.5)),
            })
        return h

    def _preds(n=100):
        y_true = rng.integers(0, 2, n).tolist()
        y_prob = np.clip(
            np.array(y_true, float) + rng.normal(0, 0.35, n), 0.01, 0.99
        ).tolist()
        threshold = dynamic_threshold(y_prob)
        y_pred = [int(p >= threshold) for p in y_prob]
        return {"y_true": y_true, "y_pred": y_pred, "y_prob": y_prob}

    demo_hist = {
        "mri_custom":        _hist(12, 0.07),
        "mri_mobilenet":     _hist(10, 0.05),
        "mri_efficientnet":  _hist(12, 0.04),
        "video":             _hist(10, 0.08),
        "scribble_custom":   _hist(12, 0.06),
        "scribble_efficientnet": _hist(12, 0.03),
        "fusion":            _hist(12, 0.03),
    }
    demo_results = {k: _preds(100) for k in demo_hist}

    report = run_full_evaluation(demo_hist, demo_results)
    print(f"All outputs saved to ./{RESULTS_DIR}/")
