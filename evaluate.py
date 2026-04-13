"""
evaluate.py — Model evaluation & visualization for NeuroScan AI
─────────────────────────────────────────────────────────────────
Generates and saves to results/:
  • confusion_matrix.png   (per model + fused)
  • accuracy_graph.png     (train vs val accuracy over epochs)
  • loss_graph.png         (train vs val loss over epochs)
  • metrics_report.json    (all scalar metrics, image paths)

Usage (standalone):
    python evaluate.py

Or import and call from train.py:
    from evaluate import run_full_evaluation
    run_full_evaluation(all_histories, all_test_results)
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for servers
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    roc_curve, auc,
)
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# ── Output directory ──────────────────────────────────────────────────────────
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Visual theme ──────────────────────────────────────────────────────────────
BG       = "#0a0e1a"
SURFACE  = "#111827"
CARD     = "#161f30"
ACCENT   = "#00e5ff"
PURPLE   = "#7c3aed"
GREEN    = "#10b981"
RED      = "#ef4444"
YELLOW   = "#f59e0b"
TEXT     = "#e2e8f0"
MUTED    = "#64748b"
BORDER   = "#1e293b"
GRID     = "#1e293b"

MODEL_COLORS = {
    "mri":      ACCENT,
    "video":    PURPLE,
    "scribble": GREEN,
    "fusion":   YELLOW,
}
MODEL_LABELS = {
    "mri":      "MRI",
    "video":    "Tremor Video",
    "scribble": "Scribble Test",
    "fusion":   "Multi-Modal Fusion",
}

def _apply_dark_theme():
    plt.rcParams.update({
        "figure.facecolor":  BG,
        "axes.facecolor":    SURFACE,
        "axes.edgecolor":    BORDER,
        "axes.labelcolor":   TEXT,
        "axes.titlecolor":   TEXT,
        "xtick.color":       MUTED,
        "ytick.color":       MUTED,
        "grid.color":        GRID,
        "grid.linewidth":    0.6,
        "text.color":        TEXT,
        "legend.facecolor":  CARD,
        "legend.edgecolor":  BORDER,
        "font.family":       "monospace",
        "font.size":         9,
        "axes.titlesize":    11,
        "axes.labelsize":    9,
    })


# ═══════════════════════════════════════════════════════════════════════════════
#  1. SCALAR METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(y_true: List[int], y_pred: List[int],
                    y_prob: Optional[List[float]] = None,
                    model_name: str = "model") -> Dict:
    """
    Compute accuracy, precision, recall, F1, confusion matrix.
    Optionally computes ROC-AUC when probabilities are provided.
    Returns a plain dict (JSON-serialisable).
    """
    y_true = list(y_true)
    y_pred = list(y_pred)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, cm[0, 0])

    metrics = {
        "model":      model_name,
        "timestamp":  datetime.now().isoformat(timespec="seconds"),
        "n_samples":  len(y_true),
        "n_positive": int(sum(y_true)),
        "n_negative": int(len(y_true) - sum(y_true)),
        "accuracy":   round(float(accuracy_score(y_true, y_pred)), 4),
        "precision":  round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall":     round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1_score":   round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "specificity":round(float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0, 4),
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp),
            "fn": int(fn), "tp": int(tp),
        },
        "classification_report": classification_report(
            y_true, y_pred,
            target_names=["No PD", "Parkinson's"],
            zero_division=0,
            output_dict=True,
        ),
    }
    if y_prob is not None:
        try:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            metrics["roc_auc"] = round(float(auc(fpr, tpr)), 4)
            metrics["_roc_fpr"] = [round(v, 4) for v in fpr.tolist()]
            metrics["_roc_tpr"] = [round(v, 4) for v in tpr.tolist()]
        except Exception:
            pass
    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
#  2. CONFUSION MATRIX PLOT
# ═══════════════════════════════════════════════════════════════════════════════

def plot_confusion_matrix(
    test_results: Dict[str, Dict],
    save_path: str = None,
) -> str:
    """
    test_results: { model_name: {"y_true": [...], "y_pred": [...]} }
    Draws one subplot per model in a grid.
    Returns the saved file path.
    """
    _apply_dark_theme()
    save_path = save_path or os.path.join(RESULTS_DIR, "confusion_matrix.png")

    models = list(test_results.keys())
    n = len(models)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4.2 * rows),
                              facecolor=BG)
    fig.suptitle("Confusion Matrices — NeuroScan AI",
                 fontsize=14, color=TEXT, fontweight="bold", y=1.01)

    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    for i, model_name in enumerate(models):
        ax = axes_flat[i]
        data = test_results[model_name]
        y_true, y_pred = data["y_true"], data["y_pred"]
        cm = confusion_matrix(y_true, y_pred)

        color = MODEL_COLORS.get(model_name, ACCENT)
        label = MODEL_LABELS.get(model_name, model_name.upper())

        # Custom colormap from dark background to model accent colour
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list(
            "neuro", [SURFACE, color], N=256
        )

        sns.heatmap(
            cm, annot=True, fmt="d", ax=ax,
            cmap=cmap,
            linewidths=1, linecolor=BORDER,
            annot_kws={"size": 16, "weight": "bold", "color": BG},
            cbar=False,
            xticklabels=["No PD", "PD"],
            yticklabels=["No PD", "PD"],
        )
        ax.set_facecolor(SURFACE)
        ax.set_title(label, color=color, fontsize=11, fontweight="bold", pad=10)
        ax.set_xlabel("Predicted", color=MUTED, fontsize=9)
        ax.set_ylabel("Actual",    color=MUTED, fontsize=9)
        ax.tick_params(colors=MUTED)

        # Annotate TP / TN / FP / FN corners
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
            acc = accuracy_score(y_true, y_pred)
            f1  = f1_score(y_true, y_pred, zero_division=0)
            ax.set_xlabel(
                f"Predicted    |   Acc: {acc:.2f}  F1: {f1:.2f}",
                color=MUTED, fontsize=8
            )

    # Hide unused subplots
    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout(pad=2.0)
    fig.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close(fig)
    print(f"[Eval] ✓ Saved → {save_path}")
    return save_path


# ═══════════════════════════════════════════════════════════════════════════════
#  3. ACCURACY GRAPH
# ═══════════════════════════════════════════════════════════════════════════════

def plot_accuracy_graph(
    histories: Dict[str, List[Dict]],
    save_path: str = None,
) -> str:
    """
    histories: { model_name: [{"epoch": 1, "acc": 0.7, ...}, ...] }
    Plots train-acc and val-acc for every model on one figure.
    """
    _apply_dark_theme()
    save_path = save_path or os.path.join(RESULTS_DIR, "accuracy_graph.png")

    models = [m for m, h in histories.items() if h]
    n = len(models)
    cols = min(n, 2)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6.5 * cols, 4.5 * rows),
                              facecolor=BG)
    fig.suptitle("Training vs Validation Accuracy", fontsize=14,
                 color=TEXT, fontweight="bold", y=1.01)

    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    for i, model_name in enumerate(models):
        ax = axes_flat[i]
        hist = histories[model_name]
        epochs     = [h["epoch"]    for h in hist]
        train_acc  = [h.get("acc", h.get("train_acc", 0)) for h in hist]
        val_acc    = [h.get("val_acc", h.get("acc", 0))   for h in hist]

        color = MODEL_COLORS.get(model_name, ACCENT)
        label = MODEL_LABELS.get(model_name, model_name.upper())

        ax.set_facecolor(SURFACE)
        ax.grid(True, alpha=0.3)

        # Filled area under train curve
        ax.fill_between(epochs, train_acc, alpha=0.08, color=color)
        ax.plot(epochs, train_acc, color=color,      linewidth=2,
                linestyle="-",  label="Train Acc",  marker="o",
                markersize=3, markevery=max(1, len(epochs)//10))
        ax.plot(epochs, val_acc,   color=MUTED,      linewidth=1.8,
                linestyle="--", label="Val Acc",    marker="s",
                markersize=3, markevery=max(1, len(epochs)//10))

        # Best val accuracy marker
        best_idx = int(np.argmax(val_acc))
        ax.scatter(epochs[best_idx], val_acc[best_idx],
                   color=color, s=80, zorder=5, edgecolors=BG, linewidth=1.5)
        ax.annotate(f"Best {val_acc[best_idx]:.3f}",
                    xy=(epochs[best_idx], val_acc[best_idx]),
                    xytext=(6, 6), textcoords="offset points",
                    color=color, fontsize=8)

        ax.set_title(label, color=color, fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch", color=MUTED)
        ax.set_ylabel("Accuracy", color=MUTED)
        ax.set_ylim(0, 1.05)
        ax.legend(loc="lower right")
        ax.spines[:].set_color(BORDER)

    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout(pad=2.0)
    fig.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close(fig)
    print(f"[Eval] ✓ Saved → {save_path}")
    return save_path


# ═══════════════════════════════════════════════════════════════════════════════
#  4. LOSS GRAPH
# ═══════════════════════════════════════════════════════════════════════════════

def plot_loss_graph(
    histories: Dict[str, List[Dict]],
    save_path: str = None,
) -> str:
    """
    histories: { model_name: [{"epoch": 1, "train_loss": 0.6, "val_loss": 0.5}, ...] }
    """
    _apply_dark_theme()
    save_path = save_path or os.path.join(RESULTS_DIR, "loss_graph.png")

    models = [m for m, h in histories.items() if h]
    n = len(models)
    cols = min(n, 2)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6.5 * cols, 4.5 * rows),
                              facecolor=BG)
    fig.suptitle("Training vs Validation Loss", fontsize=14,
                 color=TEXT, fontweight="bold", y=1.01)

    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    for i, model_name in enumerate(models):
        ax = axes_flat[i]
        hist = histories[model_name]
        epochs     = [h["epoch"]      for h in hist]
        train_loss = [h["train_loss"] for h in hist]
        val_loss   = [h["val_loss"]   for h in hist]

        color = MODEL_COLORS.get(model_name, ACCENT)
        label = MODEL_LABELS.get(model_name, model_name.upper())

        ax.set_facecolor(SURFACE)
        ax.grid(True, alpha=0.3)

        ax.fill_between(epochs, train_loss, alpha=0.08, color=color)
        ax.plot(epochs, train_loss, color=color, linewidth=2,
                linestyle="-", label="Train Loss", marker="o",
                markersize=3, markevery=max(1, len(epochs)//10))
        ax.plot(epochs, val_loss, color=RED, linewidth=1.8,
                linestyle="--", label="Val Loss", marker="s",
                markersize=3, markevery=max(1, len(epochs)//10))

        # Mark minimum val loss (best checkpoint epoch)
        best_idx = int(np.argmin(val_loss))
        ax.axvline(epochs[best_idx], color=color, linewidth=0.8,
                   linestyle=":", alpha=0.6)
        ax.scatter(epochs[best_idx], val_loss[best_idx],
                   color=RED, s=80, zorder=5, edgecolors=BG, linewidth=1.5)
        ax.annotate(f"Min {val_loss[best_idx]:.3f}",
                    xy=(epochs[best_idx], val_loss[best_idx]),
                    xytext=(6, 6), textcoords="offset points",
                    color=RED, fontsize=8)

        ax.set_title(label, color=color, fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch", color=MUTED)
        ax.set_ylabel("Loss (BCE)", color=MUTED)
        ax.legend(loc="upper right")
        ax.spines[:].set_color(BORDER)

    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout(pad=2.0)
    fig.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close(fig)
    print(f"[Eval] ✓ Saved → {save_path}")
    return save_path


# ═══════════════════════════════════════════════════════════════════════════════
#  5. METRICS SUMMARY DASHBOARD (bonus — all models side-by-side bar chart)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_metrics_dashboard(
    all_metrics: Dict[str, Dict],
    save_path: str = None,
) -> str:
    """
    all_metrics: { model_name: metrics_dict }
    Creates a compact grouped-bar chart of Acc / Prec / Recall / F1 / Spec.
    """
    _apply_dark_theme()
    save_path = save_path or os.path.join(RESULTS_DIR, "metrics_dashboard.png")

    metric_keys   = ["accuracy", "precision", "recall", "f1_score", "specificity"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score", "Specificity"]
    models        = list(all_metrics.keys())
    x             = np.arange(len(metric_keys))
    width         = 0.8 / max(len(models), 1)

    fig, ax = plt.subplots(figsize=(10, 5), facecolor=BG)
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis="y", alpha=0.3)

    for j, model_name in enumerate(models):
        m     = all_metrics[model_name]
        vals  = [m.get(k, 0.0) for k in metric_keys]
        color = MODEL_COLORS.get(model_name, ACCENT)
        label = MODEL_LABELS.get(model_name, model_name.upper())
        offset = (j - len(models) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width * 0.88, label=label,
                      color=color, alpha=0.85, edgecolor=BG, linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.012,
                    f"{val:.2f}", ha="center", va="bottom",
                    color=color, fontsize=7, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, color=TEXT)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score", color=MUTED)
    ax.set_title("Model Performance Comparison", color=TEXT,
                 fontsize=13, fontweight="bold", pad=14)
    ax.spines[:].set_color(BORDER)
    ax.legend(loc="upper right", fontsize=9)
    ax.axhline(1.0, color=BORDER, linewidth=0.7, linestyle="--")

    plt.tight_layout(pad=2.0)
    fig.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close(fig)
    print(f"[Eval] ✓ Saved → {save_path}")
    return save_path


# ═══════════════════════════════════════════════════════════════════════════════
#  6. MASTER ENTRY POINT — called from train.py
# ═══════════════════════════════════════════════════════════════════════════════

def run_full_evaluation(
    histories: Dict[str, List[Dict]],
    test_results: Dict[str, Dict],
) -> Dict:
    """
    histories:    { model_name: [epoch_dict, ...] }
    test_results: { model_name: {"y_true": [...], "y_pred": [...], "y_prob": [...]} }

    Returns the master metrics dict (also saved to results/metrics_report.json).
    """
    print("\n" + "─" * 50)
    print("  Running Evaluation & Generating Visualisations")
    print("─" * 50)

    # ── Scalar metrics ─────────────────────────────────────────────
    all_metrics = {}
    for model_name, data in test_results.items():
        m = compute_metrics(
            data["y_true"],
            data["y_pred"],
            data.get("y_prob"),
            model_name,
        )
        all_metrics[model_name] = m
        print(f"  [{MODEL_LABELS.get(model_name, model_name)}] "
              f"Acc={m['accuracy']:.3f}  Prec={m['precision']:.3f}  "
              f"Rec={m['recall']:.3f}  F1={m['f1_score']:.3f}  "
              f"Spec={m['specificity']:.3f}")

    # ── Plots ──────────────────────────────────────────────────────
    cm_path   = plot_confusion_matrix(test_results)
    acc_path  = plot_accuracy_graph(histories) if any(histories.values()) else None
    loss_path = plot_loss_graph(histories)     if any(histories.values()) else None
    dash_path = plot_metrics_dashboard(all_metrics)

    # ── JSON report ────────────────────────────────────────────────
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "models":        all_metrics,
        "image_paths": {
            "confusion_matrix":    cm_path,
            "accuracy_graph":      acc_path,
            "loss_graph":          loss_path,
            "metrics_dashboard":   dash_path,
        },
    }
    report_path = os.path.join(RESULTS_DIR, "metrics_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[Eval] ✓ JSON report → {report_path}")
    print("─" * 50 + "\n")
    return report


# ═══════════════════════════════════════════════════════════════════════════════
#  Standalone demo (synthetic data)
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    rng = np.random.default_rng(42)

    def _synthetic_history(epochs=30, noisy=0.05):
        h = []
        for e in range(1, epochs + 1):
            prog = e / epochs
            h.append({
                "epoch":      e,
                "train_loss": max(0.05, 0.65 * (1 - prog) + rng.normal(0, noisy)),
                "val_loss":   max(0.08, 0.60 * (1 - prog * 0.85) + rng.normal(0, noisy * 1.5)),
                "acc":        min(0.98, 0.50 + 0.44 * prog + rng.normal(0, noisy)),
                "val_acc":    min(0.96, 0.50 + 0.40 * prog + rng.normal(0, noisy * 1.5)),
            })
        return h

    def _synthetic_preds(n=120, acc=0.84):
        y_true = rng.integers(0, 2, n).tolist()
        y_prob = np.clip(
            np.array(y_true, float) + rng.normal(0, 0.35, n), 0.01, 0.99
        ).tolist()
        y_pred = [int(p >= 0.5) for p in y_prob]
        return {"y_true": y_true, "y_pred": y_pred, "y_prob": y_prob}

    demo_histories = {
        "mri":      _synthetic_history(30),
        "video":    _synthetic_history(20, 0.07),
        "scribble": _synthetic_history(30, 0.04),
        "fusion":   _synthetic_history(20, 0.03),
    }
    demo_results = {
        "mri":      _synthetic_preds(120, 0.82),
        "video":    _synthetic_preds(80,  0.78),
        "scribble": _synthetic_preds(120, 0.85),
        "fusion":   _synthetic_preds(120, 0.91),
    }

    report = run_full_evaluation(demo_histories, demo_results)
    print(json.dumps(
        {k: {m: v for m, v in d.items() if not m.startswith("_")}
         for k, d in report["models"].items()},
        indent=2
    ))
    print(f"\nAll outputs saved to ./{RESULTS_DIR}/")
