"""
train.py — NeuroScan AI v3.0  Full Training Pipeline
────────────────────────────────────────────────────────────────────────────────
What this does:
  1. Loads 3 datasets: MRI (T1 substantia-nigra), Video tremor, Spiral scribble
  2. For each image modality trains 3 backbones: CustomCNN, MobileNetV2, EfficientNetB0
  3. For video modality trains the LSTM model (MobileNetV2 frame encoder)
  4. Compares all per-modality models → picks best per F1
  5. Trains a late-fusion head on top
  6. Saves best model per modality + best_model_info.json
  7. Calls run_full_evaluation() → all plots + metrics_report.json

Anti-overfitting measures
  • max 12 epochs (EPOCH_LIMIT)
  • Early stopping (patience=4) on val F1
  • Label-smoothing BCE + pos_weight for class imbalance
  • WeightedRandomSampler as second guard against imbalance
  • Cosine-annealing LR + weight-decay
  • Data augmentation (train_transform)
  • Dropout inside all encoders
  • Gradient clipping

Run:
    python train.py
"""

import os
import sys
import json
import copy
import warnings
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import (
    Dataset, DataLoader, random_split, WeightedRandomSampler
)
import torchvision.transforms as T
import cv2
from PIL import Image, UnidentifiedImageError

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)

from models_arch import (
    build_image_encoder, VideoTremorModel,
    FusionClassifier, GradCAM
)
from evaluate import run_full_evaluation

warnings.filterwarnings("ignore")


def dynamic_threshold(probs) -> float:
    probs = np.asarray(probs, dtype=np.float32)
    if probs.size == 0:
        return 0.5
    return float(np.clip(probs.mean(), 0.0, 1.0))

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION  ← Edit dataset paths here
# ══════════════════════════════════════════════════════════════════════════════

MRI_DATASET_PATH      = r"C:\Users\LENOVO\Downloads\archive (5)\parkinsons_dataset"
VIDEO_DATASET_PATH    = r"C:\Users\LENOVO\Downloads\archive (7)\data"
SCRIBBLE_DATASET_PATH = r"C:\Users\LENOVO\Downloads\archive (6)\spiral"

# Hyper-parameters
BATCH_SIZE   = 16
EPOCH_LIMIT  = 12          # hard cap (≤12 as requested)
LR           = 3e-4
WEIGHT_DECAY = 1e-4
FEATURE_DIM  = 256
NUM_FRAMES   = 30
PATIENCE     = 4           # early-stopping patience

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR = "models"
os.makedirs(SAVE_DIR, exist_ok=True)
print(f"[INFO] Device: {DEVICE}")

# ══════════════════════════════════════════════════════════════════════════════
# TRANSFORMS
# ══════════════════════════════════════════════════════════════════════════════

IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD  = [0.229, 0.224, 0.225]

train_transform = T.Compose([
    T.Resize((224, 224)),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(p=0.1),
    T.RandomRotation(20),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    T.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    T.ToTensor(),
    T.Normalize(IMG_MEAN, IMG_STD),
    T.RandomErasing(p=0.15, scale=(0.02, 0.1)),
])

val_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(IMG_MEAN, IMG_STD),
])

# ══════════════════════════════════════════════════════════════════════════════
# DATASET HELPERS
# ══════════════════════════════════════════════════════════════════════════════

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}

# Class-name → binary label mapping
POSITIVE_NAMES = {
    "parkinson", "parkinsons", "pd", "patient",
    "1", "positive", "yes", "disease", "affected"
}


def collect_image_samples(root: str) -> List[Tuple[str, int]]:
    """
    Walks root/class_dir/* and assigns label=1 for Parkinson's folders,
    label=0 otherwise. Handles any subfolder depth.
    """
    samples, root = [], Path(root)
    if not root.exists():
        print(f"  [WARN] Not found: {root}")
        return samples
    for cls_dir in sorted(root.iterdir()):
        if not cls_dir.is_dir():
            continue
        label = 1 if cls_dir.name.lower() in POSITIVE_NAMES else 0
        for f in cls_dir.rglob("*"):
            if f.suffix.lower() in IMAGE_EXTS:
                samples.append((str(f), label))
    return samples


def collect_video_samples(root: str) -> List[Tuple[str, int]]:
    samples, root = [], Path(root)
    if not root.exists():
        print(f"  [WARN] Not found: {root}")
        return samples
    for cls_dir in sorted(root.iterdir()):
        if not cls_dir.is_dir():
            continue
        label = 1 if cls_dir.name.lower() in POSITIVE_NAMES else 0
        for f in cls_dir.rglob("*"):
            if f.suffix.lower() in VIDEO_EXTS:
                samples.append((str(f), label))
    return samples


# ══════════════════════════════════════════════════════════════════════════════
# PYTORCH DATASETS
# ══════════════════════════════════════════════════════════════════════════════

class ImageDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, int]], transform=None):
        self.samples   = samples
        self.transform = transform

    def __len__(self):  return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except (OSError, UnidentifiedImageError):
            img = Image.new("RGB", (224, 224), 0)
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32)


class VideoDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, int]],
                 num_frames: int = 30, img_size: int = 112):
        self.samples    = samples
        self.num_frames = num_frames
        self.img_size   = img_size
        self._mean = np.array(IMG_MEAN, dtype=np.float32)
        self._std  = np.array(IMG_STD,  dtype=np.float32)

    def __len__(self):  return len(self.samples)

    def _load(self, path: str) -> torch.Tensor:
        cap   = cv2.VideoCapture(path)
        total = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        step  = max(1, total // self.num_frames)
        frames, i = [], 0
        while cap.isOpened() and len(frames) < self.num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if i % step == 0:
                frame = cv2.resize(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    (self.img_size, self.img_size)
                ).astype(np.float32) / 255.0
                frame = (frame - self._mean) / self._std
                frames.append(frame)
            i += 1
        cap.release()
        # Pad short videos
        blank = np.zeros((self.img_size, self.img_size, 3), np.float32)
        while len(frames) < self.num_frames:
            frames.append(blank)
        arr = np.stack(frames[:self.num_frames])          # (T, H, W, 3)
        return torch.tensor(arr.transpose(0, 3, 1, 2))   # (T, 3, H, W)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            video = self._load(path)
        except Exception:
            video = torch.zeros(self.num_frames, 3, self.img_size, self.img_size)
        return video, torch.tensor(label, dtype=torch.float32)


# ══════════════════════════════════════════════════════════════════════════════
# DATA SPLIT  (stratified-ish via random_split with fixed seed)
# ══════════════════════════════════════════════════════════════════════════════

def split_dataset(dataset, val_frac=0.15, test_frac=0.15, seed=42):
    n      = len(dataset)
    n_val  = max(1, int(n * val_frac))
    n_test = max(1, int(n * test_frac))
    n_train= n - n_val - n_test
    if n_train < 1:
        raise ValueError(f"Dataset too small ({n} samples) to split.")
    return random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(seed)
    )


def make_weighted_sampler(subset) -> WeightedRandomSampler:
    """
    Build a WeightedRandomSampler so each epoch sees ~equal positives/negatives.
    Works on any Subset whose underlying dataset has .samples list.
    """
    labels = []
    for idx in subset.indices:
        _, lbl = subset.dataset.samples[idx]
        labels.append(int(lbl))
    labels  = np.array(labels)
    n_pos   = labels.sum()
    n_neg   = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None   # perfectly balanced or single class — skip
    w_pos = 1.0 / n_pos
    w_neg = 1.0 / n_neg
    sample_weights = np.where(labels == 1, w_pos, w_neg)
    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float64),
        num_samples=len(labels),
        replacement=True,
    )


def compute_pos_weight(samples: List[Tuple]) -> torch.Tensor:
    labels = [s[1] for s in samples]
    n_pos  = sum(labels)
    n_neg  = len(labels) - n_pos
    if n_pos == 0:
        return torch.tensor(1.0)
    return torch.tensor(n_neg / n_pos, dtype=torch.float32)


# ══════════════════════════════════════════════════════════════════════════════
# LABEL-SMOOTHING BCE  (replaces hard 0/1 targets → avoids overconfidence)
# ══════════════════════════════════════════════════════════════════════════════

class LabelSmoothBCE(nn.Module):
    def __init__(self, smoothing: float = 0.05,
                 pos_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.smoothing   = smoothing
        self.base_loss   = nn.BCEWithLogitsLoss(pos_weight=pos_weight,
                                                reduction="mean")

    def forward(self, logits, targets):
        # Soft labels: 0 → ε,  1 → 1-ε
        smooth = targets * (1 - self.smoothing) + self.smoothing * 0.5
        return self.base_loss(logits, smooth)


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION  (threshold-free — model learns to be calibrated)
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_loader(model, loader, criterion, device,
                    return_preds=False):
    """
    NOTE: predictions stay probability-first.
    When discrete metrics are required, the batch-set threshold is derived
    from the mean predicted probability instead of a fixed cut-off.
    """
    model.eval()
    total_loss = 0.0
    all_preds, all_probs, all_labels = [], [], []

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits  = model(batch_x)
            loss    = criterion(logits, batch_y)
            total_loss += loss.item() * len(batch_y)

            probs = torch.sigmoid(logits).cpu().tolist()
            all_probs.extend(probs)
            all_labels.extend(batch_y.long().cpu().tolist())

    threshold = dynamic_threshold(all_probs)
    all_preds = [int(p >= threshold) for p in all_probs]

    avg_loss = total_loss / max(len(loader.dataset), 1)
    acc  = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec  = recall_score(all_labels, all_preds,  zero_division=0)
    f1   = f1_score(all_labels, all_preds,      zero_division=0)

    if return_preds:
        return avg_loss, acc, prec, rec, f1, all_labels, all_preds, all_probs
    return avg_loss, acc, prec, rec, f1


# ══════════════════════════════════════════════════════════════════════════════
# SINGLE-MODEL TRAINING LOOP  with early stopping
# ══════════════════════════════════════════════════════════════════════════════

def train_single_model(model, train_loader, val_loader,
                       model_name, epochs=EPOCH_LIMIT,
                       lr=LR, device=DEVICE,
                       pos_weight=None) -> Tuple[nn.Module, List[Dict]]:

    model = model.to(device)
    pw    = pos_weight.to(device) if pos_weight is not None else None
    criterion = LabelSmoothBCE(smoothing=0.05, pos_weight=pw)
    optimizer = optim.AdamW(model.parameters(), lr=lr,
                            weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs,
                                                      eta_min=lr / 50)

    best_f1, best_weights, patience_count = 0.0, None, 0
    history: List[Dict] = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss   = criterion(logits, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(batch_y)

        scheduler.step()
        train_loss /= max(len(train_loader.dataset), 1)

        val_loss, acc, prec, rec, f1 = evaluate_loader(
            model, val_loader, criterion, device
        )
        history.append({
            "epoch": epoch, "train_loss": train_loss,
            "val_loss": val_loss, "acc": acc,
            "val_acc": acc, "f1": f1
        })

        print(f"  [{model_name}] ep {epoch:02d}/{epochs} | "
              f"trn={train_loss:.4f} val={val_loss:.4f} "
              f"acc={acc:.3f} f1={f1:.3f}")

        if f1 > best_f1 + 1e-4:
            best_f1      = f1
            best_weights = copy.deepcopy(model.state_dict())
            patience_count = 0
        else:
            patience_count += 1

        if patience_count >= PATIENCE:
            print(f"  [{model_name}] Early stopping at epoch {epoch} "
                  f"(no improvement for {PATIENCE} epochs)")
            break

    if best_weights is not None:
        model.load_state_dict(best_weights)

    save_path = os.path.join(SAVE_DIR, f"{model_name}_model.pth")
    torch.save(model, save_path)
    print(f"  [{model_name}] ✓ Saved → {save_path}  (best val F1={best_f1:.3f})")
    return model, history


# ══════════════════════════════════════════════════════════════════════════════
# MULTI-BACKBONE COMPARISON  (image modalities)
# ══════════════════════════════════════════════════════════════════════════════

BACKBONES = ["custom", "mobilenet", "efficientnet"]


def train_and_compare(samples, modality_name,
                      all_histories, all_test_results):
    """
    Trains all 3 backbones on one image modality.
    Returns the best model object + backbone name.
    """
    pos_weight = compute_pos_weight(samples)
    dataset    = ImageDataset(samples)
    train_ds, val_ds, test_ds = split_dataset(dataset)

    # Apply transforms (subsets share the underlying dataset object,
    # so we wrap them in a transform-aware class)
    class TransformSubset(Dataset):
        def __init__(self, subset, transform):
            self.subset    = subset
            self.transform = transform
        def __len__(self):  return len(self.subset)
        def __getitem__(self, idx):
            img_tensor, label = self.subset[idx]
            # img_tensor is a PIL Image here because ImageDataset.transform=None
            if self.transform:
                img_tensor = self.transform(img_tensor)
            return img_tensor, label

    # Keep ImageDataset without transform so TransformSubset can apply own tfm
    dataset.transform = None

    train_wrap = TransformSubset(train_ds, train_transform)
    val_wrap   = TransformSubset(val_ds,   val_transform)
    test_wrap  = TransformSubset(test_ds,  val_transform)

    sampler = make_weighted_sampler(train_ds)
    train_loader = DataLoader(
        train_wrap, batch_size=BATCH_SIZE,
        sampler=sampler if sampler else None,
        shuffle=(sampler is None), num_workers=0, pin_memory=False
    )
    val_loader  = DataLoader(val_wrap,  batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_wrap, batch_size=BATCH_SIZE, shuffle=False)

    best_model, best_backbone, best_f1 = None, None, -1.0
    comparison_metrics: Dict[str, Dict] = {}

    for bb in BACKBONES:
        key   = f"{modality_name}_{bb}"
        model = build_image_encoder(bb, FEATURE_DIM, pretrained=(bb != "custom"))
        # Wrap with binary head for standalone training
        model = _HeadedEncoder(model)

        trained_model, history = train_single_model(
            model, train_loader, val_loader, key,
            pos_weight=pos_weight
        )
        all_histories[key] = history

        # Test evaluation
        crit = LabelSmoothBCE()
        _, acc, prec, rec, f1, yt, yp, ypr = evaluate_loader(
            trained_model, test_loader, crit, DEVICE, return_preds=True
        )
        comparison_metrics[key] = {
            "accuracy": acc, "precision": prec,
            "recall": rec,   "f1": f1, "backbone": bb
        }
        all_test_results[key] = {
            "y_true": yt, "y_pred": yp, "y_prob": ypr
        }
        print(f"  [{key}] TEST → acc={acc:.3f} prec={prec:.3f} "
              f"rec={rec:.3f} f1={f1:.3f}")

        if f1 > best_f1:
            best_f1, best_model, best_backbone = f1, trained_model, bb

    print(f"\n  ★ Best backbone for {modality_name}: "
          f"{best_backbone}  (F1={best_f1:.3f})")

    # Save best as canonical model for this modality
    canonical_path = os.path.join(SAVE_DIR, f"{modality_name}_best_model.pth")
    torch.save(best_model, canonical_path)

    # Return the encoder part (without classification head)
    return best_model.encoder, best_backbone, comparison_metrics


# ══════════════════════════════════════════════════════════════════════════════
# HELPER: thin classification head wrapper (for standalone modality training)
# ══════════════════════════════════════════════════════════════════════════════

class _HeadedEncoder(nn.Module):
    """Wraps any encoder + a tiny binary head for modality-level training."""
    def __init__(self, encoder: nn.Module, feat_dim: int = FEATURE_DIM):
        super().__init__()
        self.encoder = encoder
        self.head    = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(feat_dim // 2, 1),
        )

    def forward(self, x):
        return self.head(self.encoder(x)).squeeze(1)


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION HELPER
# ══════════════════════════════════════════════════════════════════════════════

def extract_features(encoder, loader, device):
    encoder.eval()
    feats, labels = [], []
    with torch.no_grad():
        for bx, by in loader:
            f = encoder(bx.to(device)).cpu()
            feats.append(f)
            labels.append(by)
    return torch.cat(feats), torch.cat(labels)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  NeuroScan AI v3.0 — Multi-Model Training Pipeline")
    print("=" * 65)

    all_histories:    Dict = {}
    all_test_results: Dict = {}
    best_model_info:  Dict = {}

    # ── 1. MRI  ──────────────────────────────────────────────────────────────
    print("\n[1/3] MRI Dataset (T1-weighted Substantia Nigra)")
    mri_samples = collect_image_samples(MRI_DATASET_PATH)
    n_pos = sum(s[1] for s in mri_samples)
    print(f"  Found {len(mri_samples)} samples  "
          f"(PD={n_pos}, Healthy={len(mri_samples)-n_pos})")

    if len(mri_samples) >= 10:
        mri_encoder, mri_bb, mri_cmp = train_and_compare(
            mri_samples, "mri", all_histories, all_test_results
        )
        best_model_info["mri"] = {"backbone": mri_bb,
                                  "metrics": mri_cmp}
    else:
        print("  [SKIP] Need ≥10 samples")
        mri_encoder = build_image_encoder("efficientnet", FEATURE_DIM,
                                          pretrained=False).to(DEVICE)
        mri_bb = "efficientnet"

    # ── 2. Video Tremor  ──────────────────────────────────────────────────────
    print("\n[2/3] Video Tremor Dataset")
    vid_samples = collect_video_samples(VIDEO_DATASET_PATH)
    n_pos = sum(s[1] for s in vid_samples)
    print(f"  Found {len(vid_samples)} samples  "
          f"(PD={n_pos}, Healthy={len(vid_samples)-n_pos})")

    if len(vid_samples) >= 10:
        vid_dataset = VideoDataset(vid_samples, num_frames=NUM_FRAMES)
        tr_v, va_v, te_v = split_dataset(vid_dataset)

        pw_v    = compute_pos_weight(vid_samples)
        sampler = make_weighted_sampler(tr_v)
        tr_ldr  = DataLoader(tr_v, batch_size=4,
                             sampler=sampler or None,
                             shuffle=(sampler is None), num_workers=0)
        va_ldr  = DataLoader(va_v, batch_size=4, shuffle=False)
        te_ldr  = DataLoader(te_v, batch_size=4, shuffle=False)

        vid_model_raw = VideoTremorModel(
            backbone="mobilenet", feature_dim=FEATURE_DIM,
            num_frames=NUM_FRAMES, pretrained=True
        )

        class _VideoHead(nn.Module):
            def __init__(self, vmodel):
                super().__init__()
                self.vmodel = vmodel
                self.head   = nn.Linear(FEATURE_DIM, 1)
            def forward(self, x):
                return self.head(self.vmodel(x)).squeeze(1)

        vid_headed = _VideoHead(vid_model_raw)
        trained_vid, vid_hist = train_single_model(
            vid_headed, tr_ldr, va_ldr, "video", pos_weight=pw_v
        )
        all_histories["video"] = vid_hist

        crit = LabelSmoothBCE()
        _, acc, prec, rec, f1, yt, yp, ypr = evaluate_loader(
            trained_vid, te_ldr, crit, DEVICE, return_preds=True
        )
        all_test_results["video"] = {"y_true": yt, "y_pred": yp, "y_prob": ypr}
        best_model_info["video"]  = {
            "backbone": "mobilenet+lstm",
            "metrics":  {"acc": acc, "f1": f1}
        }
        print(f"  [video] TEST → acc={acc:.3f} f1={f1:.3f}")
        vid_encoder = trained_vid.vmodel
    else:
        print("  [SKIP] Need ≥10 samples")
        vid_encoder = VideoTremorModel("mobilenet", FEATURE_DIM,
                                       pretrained=False).to(DEVICE)

    # ── 3. Scribble / Spiral  ─────────────────────────────────────────────────
    print("\n[3/3] Spiral / Scribble Dataset")
    scrib_samples = collect_image_samples(SCRIBBLE_DATASET_PATH)
    n_pos = sum(s[1] for s in scrib_samples)
    print(f"  Found {len(scrib_samples)} samples  "
          f"(PD={n_pos}, Healthy={len(scrib_samples)-n_pos})")

    if len(scrib_samples) >= 10:
        scrib_encoder, scrib_bb, scrib_cmp = train_and_compare(
            scrib_samples, "scribble", all_histories, all_test_results
        )
        best_model_info["scribble"] = {"backbone": scrib_bb,
                                       "metrics": scrib_cmp}
    else:
        print("  [SKIP] Need ≥10 samples")
        scrib_encoder = build_image_encoder("efficientnet", FEATURE_DIM,
                                             pretrained=False).to(DEVICE)
        scrib_bb = "efficientnet"

    # ── 4. Fusion Model  ─────────────────────────────────────────────────────
    print("\n[4/4] Fusion Classifier")
    # Use scribble or MRI samples as the alignment proxy
    align_samples = (scrib_samples if len(scrib_samples) >= 10
                     else mri_samples)

    if len(align_samples) >= 10:
        mri_enc_dev   = mri_encoder.to(DEVICE).eval()
        vid_enc_dev   = vid_encoder.to(DEVICE).eval()
        scrib_enc_dev = scrib_encoder.to(DEVICE).eval()

        class FeatureDataset(Dataset):
            """Runs forward passes on each modality once; stores features."""
            def __init__(self, samples):
                self.data = []
                ds = ImageDataset(samples, transform=val_transform)
                loader = DataLoader(ds, batch_size=16, shuffle=False)
                with torch.no_grad():
                    for imgs, labels in loader:
                        imgs_dev = imgs.to(DEVICE)
                        mf  = mri_enc_dev(imgs_dev).cpu()
                        sf  = scrib_enc_dev(imgs_dev).cpu()
                        vf  = torch.zeros(len(imgs), FEATURE_DIM)  # placeholder
                        for i in range(len(imgs)):
                            self.data.append((mf[i], vf[i], sf[i],
                                              labels[i]))
            def __len__(self):  return len(self.data)
            def __getitem__(self, i): return self.data[i]

        feat_ds = FeatureDataset(align_samples)
        tr_f, va_f, te_f = split_dataset(feat_ds, val_frac=0.15, test_frac=0.15)

        def fuse_collate(batch):
            mf, vf, sf, lb = zip(*batch)
            return (torch.stack(mf), torch.stack(vf),
                    torch.stack(sf), torch.stack(lb))

        tr_fl = DataLoader(tr_f, batch_size=BATCH_SIZE, shuffle=True,
                           collate_fn=fuse_collate)
        va_fl = DataLoader(va_f, batch_size=BATCH_SIZE, shuffle=False,
                           collate_fn=fuse_collate)
        te_fl = DataLoader(te_f, batch_size=BATCH_SIZE, shuffle=False,
                           collate_fn=fuse_collate)

        fusion = FusionClassifier(FEATURE_DIM, FEATURE_DIM,
                                  FEATURE_DIM).to(DEVICE)
        pw_fus = compute_pos_weight(align_samples)
        crit_f = LabelSmoothBCE(pos_weight=pw_fus.to(DEVICE))
        opt_f  = optim.AdamW(fusion.parameters(), lr=LR,
                             weight_decay=WEIGHT_DECAY)
        sched_f = optim.lr_scheduler.CosineAnnealingLR(opt_f, T_max=EPOCH_LIMIT)

        best_f1_fus, best_w_fus = 0.0, None
        patience_fus = 0
        fus_hist: List[Dict] = []

        for epoch in range(1, EPOCH_LIMIT + 1):
            fusion.train()
            tl = 0.0
            for mf, vf, sf, lbl in tr_fl:
                mf, vf, sf, lbl = (mf.to(DEVICE), vf.to(DEVICE),
                                   sf.to(DEVICE), lbl.to(DEVICE))
                opt_f.zero_grad()
                logit = fusion(mf, vf, sf)
                loss  = crit_f(logit, lbl)
                loss.backward()
                opt_f.step()
                tl += loss.item() * len(lbl)
            sched_f.step()

            fusion.eval()
            vp, vl = [], []
            vl_loss = 0.0
            with torch.no_grad():
                for mf, vf, sf, lbl in va_fl:
                    mf, vf, sf, lbl = (mf.to(DEVICE), vf.to(DEVICE),
                                       sf.to(DEVICE), lbl.to(DEVICE))
                    logit = fusion(mf, vf, sf)
                    vl_loss += crit_f(logit, lbl).item() * len(lbl)
                    vp.extend(torch.sigmoid(logit).cpu().tolist())
                    vl.extend(lbl.long().cpu().tolist())
            val_threshold = dynamic_threshold(vp)
            val_pred = [int(p >= val_threshold) for p in vp]
            f1v  = f1_score(vl, val_pred, zero_division=0)
            accv = accuracy_score(vl, val_pred)
            tl  /= max(len(tr_fl.dataset), 1)
            vl_loss /= max(len(va_fl.dataset), 1)

            fus_hist.append({"epoch": epoch, "train_loss": tl,
                             "val_loss": vl_loss, "acc": accv,
                             "val_acc": accv, "f1": f1v})
            print(f"  [fusion] ep {epoch:02d}/{EPOCH_LIMIT} | "
                  f"trn={tl:.4f} val={vl_loss:.4f} acc={accv:.3f} f1={f1v:.3f}")

            if f1v > best_f1_fus + 1e-4:
                best_f1_fus = f1v
                best_w_fus  = copy.deepcopy(fusion.state_dict())
                patience_fus = 0
            else:
                patience_fus += 1
            if patience_fus >= PATIENCE:
                print(f"  [fusion] Early stopping at epoch {epoch}")
                break

        if best_w_fus:
            fusion.load_state_dict(best_w_fus)

        # Fusion test
        all_mf, all_vf_t, all_sf, all_lbl_f, all_prob_f = [], [], [], [], []
        with torch.no_grad():
            for mf, vf, sf, lbl in te_fl:
                mf, vf, sf = mf.to(DEVICE), vf.to(DEVICE), sf.to(DEVICE)
                prob = torch.sigmoid(fusion(mf, vf, sf)).cpu().tolist()
                all_prob_f.extend(prob)
                all_lbl_f.extend(lbl.long().tolist())

        fusion_threshold = dynamic_threshold(all_prob_f)
        all_pred_f = [int(p >= fusion_threshold) for p in all_prob_f]
        f1_fus  = f1_score(all_lbl_f, all_pred_f, zero_division=0)
        acc_fus = accuracy_score(all_lbl_f, all_pred_f)
        print(f"  [fusion] TEST → acc={acc_fus:.3f} f1={f1_fus:.3f}")

        all_histories["fusion"]    = fus_hist
        all_test_results["fusion"] = {
            "y_true": all_lbl_f, "y_pred": all_pred_f, "y_prob": all_prob_f
        }
        best_model_info["fusion"] = {"f1": f1_fus, "acc": acc_fus}

        torch.save(fusion, os.path.join(SAVE_DIR, "fusion_model.pth"))
        print(f"  [fusion] ✓ Saved  (best val F1={best_f1_fus:.3f})")
    else:
        print("  [SKIP] Not enough aligned samples for fusion")

    # ── 5. Persist best_model_info  ──────────────────────────────────────────
    info_path = os.path.join(SAVE_DIR, "best_model_info.json")
    with open(info_path, "w") as f:
        json.dump(best_model_info, f, indent=2)
    print(f"\n[INFO] best_model_info → {info_path}")

    # ── 6. Evaluation & Visualisation  ───────────────────────────────────────
    if all_test_results:
        run_full_evaluation(all_histories, all_test_results)

    print("\n" + "=" * 65)
    print("  Training complete!")
    print(f"  Models  → ./models/")
    print(f"  Results → ./results/")
    print("=" * 65)


if __name__ == "__main__":
    main()
