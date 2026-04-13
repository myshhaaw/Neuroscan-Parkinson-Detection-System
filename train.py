"""
train.py — Full training pipeline for NeuroScan AI
Run: python train.py
"""

import os, sys, json, time, copy
import numpy as np
from pathlib import Path
from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
import cv2
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from models_arch import MRIModel, VideoTremorModel, ScribbleModel, FusionClassifier
from evaluate import run_full_evaluation

# ═══════════════════════════════════════════════════════════════════════════════
#  DATASET PATHS  ←  Edit these to match your local folder structure
# ═══════════════════════════════════════════════════════════════════════════════

MRI_DATASET_PATH     = r"C:\Users\LENOVO\Downloads\archive (5)\parkinsons_dataset"
VIDEO_DATASET_PATH   = r"C:\Users\LENOVO\Downloads\archive (7)\data"
SCRIBBLE_DATASET_PATH= r"C:\Users\LENOVO\Downloads\archive (6)\spiral"

# ═══════════════════════════════════════════════════════════════════════════════
#  HYPERPARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════

BATCH_SIZE    = 16
NUM_EPOCHS    = 30
LR            = 1e-4
WEIGHT_DECAY  = 1e-4
FEATURE_DIM   = 256
NUM_FRAMES    = 30
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR      = "models"

os.makedirs(SAVE_DIR, exist_ok=True)
print(f"[INFO] Using device: {DEVICE}")


# ═══════════════════════════════════════════════════════════════════════════════
#  TRANSFORMS
# ═══════════════════════════════════════════════════════════════════════════════

IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD  = [0.229, 0.224, 0.225]

train_transform = T.Compose([
    T.Resize((224, 224)),
    T.RandomHorizontalFlip(),
    T.RandomRotation(15),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    T.ToTensor(),
    T.Normalize(IMG_MEAN, IMG_STD),
])

val_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(IMG_MEAN, IMG_STD),
])


# ═══════════════════════════════════════════════════════════════════════════════
#  DATASETS
# ═══════════════════════════════════════════════════════════════════════════════

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def collect_image_samples(root: str):
    """
    Expects: root/healthy/* and root/parkinson/* (or root/0/* and root/1/*).
    Returns list of (path, label) tuples where label ∈ {0, 1}.
    """
    POSITIVE_NAMES = {"parkinson", "parkinsons", "pd", "patient", "1", "positive", "yes"}
    samples = []
    root = Path(root)
    if not root.exists():
        print(f"[WARN] Path not found: {root}")
        return samples
    for class_dir in sorted(root.iterdir()):
        if not class_dir.is_dir():
            continue
        label = 1 if class_dir.name.lower() in POSITIVE_NAMES else 0
        for f in class_dir.rglob("*"):
            if f.suffix.lower() in IMAGE_EXTS:
                samples.append((str(f), label))
    return samples


def collect_video_samples(root: str):
    """Same folder convention as images but for video files."""
    POSITIVE_NAMES = {"parkinson", "parkinsons", "pd", "patient", "1", "positive", "yes"}
    samples = []
    root = Path(root)
    if not root.exists():
        print(f"[WARN] Path not found: {root}")
        return samples
    for class_dir in sorted(root.iterdir()):
        if not class_dir.is_dir():
            continue
        label = 1 if class_dir.name.lower() in POSITIVE_NAMES else 0
        for f in class_dir.rglob("*"):
            if f.suffix.lower() in VIDEO_EXTS:
                samples.append((str(f), label))
    return samples


class ImageDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples   = samples
        self.transform = transform

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), 0)
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32)


class VideoDataset(Dataset):
    def __init__(self, samples, num_frames=30, img_size=112):
        self.samples    = samples
        self.num_frames = num_frames
        self.img_size   = img_size
        self.norm_mean  = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.norm_std   = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self): return len(self.samples)

    def _load_video(self, path):
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        step = max(1, total // self.num_frames)
        frames, idx = [], 0
        while cap.isOpened() and len(frames) < self.num_frames:
            ret, frame = cap.read()
            if not ret: break
            if idx % step == 0:
                frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                                   (self.img_size, self.img_size))
                frame = frame.astype(np.float32) / 255.0
                frame = (frame - self.norm_mean) / self.norm_std
                frames.append(frame)
            idx += 1
        cap.release()
        while len(frames) < self.num_frames:
            frames.append(np.zeros((self.img_size, self.img_size, 3), np.float32))
        frames = np.stack(frames[:self.num_frames])  # (T, H, W, 3)
        return torch.tensor(frames.transpose(0, 3, 1, 2))  # (T, 3, H, W)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            video = self._load_video(path)
        except Exception:
            video = torch.zeros(self.num_frames, 3, self.img_size, self.img_size)
        return video, torch.tensor(label, dtype=torch.float32)


# ═══════════════════════════════════════════════════════════════════════════════
#  TRAINING HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def split_dataset(dataset, val_frac=0.15, test_frac=0.15, seed=42):
    n = len(dataset)
    n_val  = int(n * val_frac)
    n_test = int(n * test_frac)
    n_train= n - n_val - n_test
    return random_split(dataset, [n_train, n_val, n_test],
                        generator=torch.Generator().manual_seed(seed))


def compute_pos_weight(samples):
    """Compute class weight for imbalanced datasets."""
    labels = [s[1] for s in samples]
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0: return torch.tensor(1.0)
    return torch.tensor(n_neg / n_pos, dtype=torch.float32)


def evaluate(model, loader, criterion, device, return_preds=False):
    model.eval()
    total_loss, all_preds, all_probs, all_labels = 0.0, [], [], []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits = model(batch_x)
            loss   = criterion(logits, batch_y)
            total_loss += loss.item() * len(batch_y)
            probs = torch.sigmoid(logits).cpu().tolist()
            preds = [int(p >= 0.5) for p in probs]
            all_preds.extend(preds)
            all_probs.extend(probs)
            all_labels.extend(batch_y.long().cpu().tolist())
    avg_loss = total_loss / len(loader.dataset)
    acc  = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec  = recall_score(all_labels, all_preds, zero_division=0)
    f1   = f1_score(all_labels, all_preds, zero_division=0)
    if return_preds:
        return avg_loss, acc, prec, rec, f1, all_labels, all_preds, all_probs
    return avg_loss, acc, prec, rec, f1


def train_single_model(model, train_loader, val_loader, model_name,
                       epochs=NUM_EPOCHS, lr=LR, device=DEVICE, pos_weight=None):
    """Generic training loop for each unimodal model."""
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device) if pos_weight is not None else None)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_f1, best_weights = 0.0, None
    history = []

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

        train_loss /= len(train_loader.dataset)
        val_loss, acc, prec, rec, f1 = evaluate(model, val_loader, criterion, device)

        print(f"[{model_name}] Epoch {epoch:02d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Acc: {acc:.3f} | Prec: {prec:.3f} | Rec: {rec:.3f} | F1: {f1:.3f}")

        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_loss": val_loss, "acc": acc, "f1": f1})

        if f1 >= best_val_f1:
            best_val_f1  = f1
            best_weights = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_weights)
    save_path = os.path.join(SAVE_DIR, f"{model_name}_model.pth")
    torch.save(model, save_path)
    print(f"[{model_name}] ✓ Best model saved → {save_path}  (Val F1={best_val_f1:.3f})")
    return model, history


# ═══════════════════════════════════════════════════════════════════════════════
#  FEATURE EXTRACTION (to build fusion training set)
# ═══════════════════════════════════════════════════════════════════════════════

def extract_features(model, loader, device):
    model.eval()
    all_feats, all_labels = [], []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            feats = model(batch_x.to(device)).cpu()
            all_feats.append(feats)
            all_labels.append(batch_y)
    return torch.cat(all_feats), torch.cat(all_labels)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN TRAINING ORCHESTRATION
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  NeuroScan AI — Training Pipeline")
    print("=" * 60)

    # Containers filled by each block; consumed by run_full_evaluation()
    all_histories:    dict = {}
    all_test_results: dict = {}

    # ── 1. MRI Model ────────────────────────────────────────────
    print("\n[1/3] Training MRI Model…")
    mri_samples = collect_image_samples(MRI_DATASET_PATH)
    print(f"      Found {len(mri_samples)} MRI samples")
    if len(mri_samples) >= 10:
        mri_dataset = ImageDataset(mri_samples)
        train_ds, val_ds, test_ds = split_dataset(mri_dataset)
        train_ds.dataset.transform = train_transform
        val_ds.dataset.transform   = val_transform
        test_ds.dataset.transform  = val_transform

        train_ldr = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
        val_ldr   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
        test_ldr  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

        pw = compute_pos_weight(mri_samples)
        mri_model, mri_history = train_single_model(
            MRIModel(FEATURE_DIM), train_ldr, val_ldr, "mri", pos_weight=pw
        )
        _, acc, prec, rec, f1, yt, yp, ypr = evaluate(
            mri_model, test_ldr, nn.BCEWithLogitsLoss(), DEVICE, return_preds=True)
        print(f"[MRI TEST]  Acc={acc:.3f}  Prec={prec:.3f}  Rec={rec:.3f}  F1={f1:.3f}")
        all_histories["mri"] = mri_history
        all_test_results["mri"] = {"y_true": yt, "y_pred": yp, "y_prob": ypr}
    else:
        print("  [SKIP] Not enough MRI samples (need ≥10)")
        mri_model = MRIModel(FEATURE_DIM, pretrained=False).to(DEVICE)

    # ── 2. Video / Tremor Model ──────────────────────────────────
    print("\n[2/3] Training Video Tremor Model…")
    vid_samples = collect_video_samples(VIDEO_DATASET_PATH)
    print(f"      Found {len(vid_samples)} video samples")
    if len(vid_samples) >= 10:
        vid_dataset = VideoDataset(vid_samples, num_frames=NUM_FRAMES)
        train_ds, val_ds, test_ds = split_dataset(vid_dataset)

        train_ldr = DataLoader(train_ds, batch_size=4, shuffle=True,  num_workers=2)
        val_ldr   = DataLoader(val_ds,   batch_size=4, shuffle=False, num_workers=2)
        test_ldr  = DataLoader(test_ds,  batch_size=4, shuffle=False, num_workers=2)

        pw = compute_pos_weight(vid_samples)
        video_model, vid_history = train_single_model(
            VideoTremorModel(FEATURE_DIM, NUM_FRAMES), train_ldr, val_ldr,
            "video", epochs=20, pos_weight=pw
        )
        _, acc, prec, rec, f1, yt, yp, ypr = evaluate(
            video_model, test_ldr, nn.BCEWithLogitsLoss(), DEVICE, return_preds=True)
        print(f"[VID TEST]  Acc={acc:.3f}  Prec={prec:.3f}  Rec={rec:.3f}  F1={f1:.3f}")
        all_histories["video"] = vid_history
        all_test_results["video"] = {"y_true": yt, "y_pred": yp, "y_prob": ypr}
    else:
        print("  [SKIP] Not enough video samples (need ≥10)")
        video_model = VideoTremorModel(FEATURE_DIM, pretrained=False).to(DEVICE)

    # ── 3. Scribble / Handwriting Model ─────────────────────────
    print("\n[3/3] Training Scribble Model…")
    scribble_samples = collect_image_samples(SCRIBBLE_DATASET_PATH)
    print(f"      Found {len(scribble_samples)} scribble samples")
    if len(scribble_samples) >= 10:
        scrib_dataset = ImageDataset(scribble_samples)
        train_ds, val_ds, test_ds = split_dataset(scrib_dataset)
        train_ds.dataset.transform = train_transform
        val_ds.dataset.transform   = val_transform
        test_ds.dataset.transform  = val_transform

        train_ldr = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
        val_ldr   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
        test_ldr  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

        pw = compute_pos_weight(scribble_samples)
        scribble_model, scrib_history = train_single_model(
            ScribbleModel(FEATURE_DIM), train_ldr, val_ldr, "scribble", pos_weight=pw
        )
        _, acc, prec, rec, f1, yt, yp, ypr = evaluate(
            scribble_model, test_ldr, nn.BCEWithLogitsLoss(), DEVICE, return_preds=True)
        print(f"[SCRIB TEST]  Acc={acc:.3f}  Prec={prec:.3f}  Rec={rec:.3f}  F1={f1:.3f}")
        all_histories["scribble"] = scrib_history
        all_test_results["scribble"] = {"y_true": yt, "y_pred": yp, "y_prob": ypr}
    else:
        print("  [SKIP] Not enough scribble samples (need ≥10)")
        scribble_model = ScribbleModel(FEATURE_DIM, pretrained=False).to(DEVICE)

    # ── 4. Fusion Model (trained on extracted features) ──────────
    print("\n[4/4] Training Fusion Classifier…")
    # For fusion training we need aligned samples.
    # Here we extract features from the scribble validation set as a proxy.
    # In production: create aligned triplet dataset.
    fusion_samples = scribble_samples if len(scribble_samples) >= 10 else mri_samples
    if len(fusion_samples) >= 10:
        class TripleDataset(Dataset):
            """Returns (mri_feat, video_feat, scrib_feat, label) by running forward passes."""
            def __init__(self, samples, mri_m, vid_m, scrib_m, transform, device):
                self.samples  = samples
                self.mri_m    = mri_m
                self.vid_m    = vid_m
                self.scrib_m  = scrib_m
                self.transform= transform
                self.device   = device

            def __len__(self): return len(self.samples)

            def __getitem__(self, idx):
                path, label = self.samples[idx]
                try:
                    img = Image.open(path).convert("RGB")
                except Exception:
                    img = Image.new("RGB", (224, 224), 0)
                t = self.transform(img).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    mf = self.mri_m(t).squeeze(0).cpu()
                    sf = self.scrib_m(t).squeeze(0).cpu()
                    vf = torch.zeros(FEATURE_DIM)  # placeholder for video
                return mf, vf, sf, torch.tensor(label, dtype=torch.float32)

        triple_ds = TripleDataset(fusion_samples, mri_model.to(DEVICE),
                                  video_model.to(DEVICE), scribble_model.to(DEVICE),
                                  val_transform, DEVICE)
        tr, va, te = split_dataset(triple_ds)
        def fuse_collate(batch):
            mf, vf, sf, lbl = zip(*batch)
            return torch.stack(mf), torch.stack(vf), torch.stack(sf), torch.stack(lbl)

        tr_ldr = DataLoader(tr, batch_size=BATCH_SIZE, shuffle=True, collate_fn=fuse_collate)
        va_ldr = DataLoader(va, batch_size=BATCH_SIZE, shuffle=False, collate_fn=fuse_collate)
        te_ldr = DataLoader(te, batch_size=BATCH_SIZE, shuffle=False, collate_fn=fuse_collate)

        fusion = FusionClassifier(FEATURE_DIM, FEATURE_DIM, FEATURE_DIM).to(DEVICE)
        criterion = nn.BCEWithLogitsLoss()
        opt = optim.AdamW(fusion.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        best_f1, best_w = 0.0, None

        for epoch in range(1, 21):
            fusion.train()
            for mf, vf, sf, lbl in tr_ldr:
                mf, vf, sf, lbl = mf.to(DEVICE), vf.to(DEVICE), sf.to(DEVICE), lbl.to(DEVICE)
                opt.zero_grad()
                logit = fusion(mf, vf, sf)
                loss  = criterion(logit, lbl)
                loss.backward()
                opt.step()

            # quick val
            fusion.eval()
            all_p, all_l = [], []
            with torch.no_grad():
                for mf, vf, sf, lbl in va_ldr:
                    mf, vf, sf = mf.to(DEVICE), vf.to(DEVICE), sf.to(DEVICE)
                    p = (torch.sigmoid(fusion(mf, vf, sf)) >= 0.5).long().cpu().tolist()
                    all_p.extend(p)
                    all_l.extend(lbl.long().tolist())
            f1v = f1_score(all_l, all_p, zero_division=0)
            print(f"[Fusion] Epoch {epoch:02d}/20 | Val F1={f1v:.3f}")
            if f1v >= best_f1:
                best_f1, best_w = f1v, copy.deepcopy(fusion.state_dict())

        fusion.load_state_dict(best_w)
        fpath = os.path.join(SAVE_DIR, "fusion_model.pth")
        torch.save(fusion, fpath)
        print(f"[Fusion] ✓ Saved → {fpath}  (Val F1={best_f1:.3f})")
    else:
        print("  [SKIP] Not enough samples for fusion training")

    # ── 5. Evaluation & Visualisation ─────────────────────
    if all_test_results:
        run_full_evaluation(all_histories, all_test_results)

    print("\n" + "="*60)
    print("  Training complete! Models saved to ./models/")
    print("  Results saved to  ./results/")
    print("  API:  uvicorn api_server:app --port 8000")
    print("  App:  streamlit run app.py")
    print("="*60)


if __name__ == "__main__":
    main()
