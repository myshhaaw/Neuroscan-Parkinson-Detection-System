"""
preprocessing.py — Dataset loading, validation, and transforms for NeuroScan AI

HANDLES THREE DATASET TYPES:
  1. MRI     — folder of .jpg/.png brain scan images
               Expected structure: <root>/healthy/   and  <root>/parkinson/
  2. Spiral  — folder of spiral drawing images (same structure as MRI)
  3. Video   — folder of .mp4/.avi tremor recordings (same folder structure)

KEY FEATURES:
  - Robust input validation (bad paths, corrupt files, wrong types)
  - Label auto-detection from folder names (no manual annotation needed)
  - Class-imbalance weight computation
  - Transforms: train (augmented) and val/test (clean)
  - VideoDataset: uniformly samples N frames per video
  - All errors are caught; corrupt samples are replaced with zeros (never crash)
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import imghdr                  # stdlib: checks actual file magic bytes

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
from PIL import Image, UnidentifiedImageError

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

# Folder-name → label mapping (0 = Healthy, 1 = Parkinson's)
POSITIVE_NAMES = {
    "parkinson", "parkinsons", "pd", "patient", "positive",
    "1", "yes", "disease", "affected", "tremor"
}
NEGATIVE_NAMES = {
    "healthy", "control", "normal", "negative", "0",
    "no", "well", "unaffected"
}

IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD  = [0.229, 0.224, 0.225]


# ─────────────────────────────────────────────────────────────────────────────
# TRANSFORMS
# ─────────────────────────────────────────────────────────────────────────────
def get_transforms(split: str = "train"):
    """
    split: "train" | "val" | "test"
    Returns a torchvision Compose transform pipeline.
    """
    if split == "train":
        return T.Compose([
            T.Resize((236, 236)),           # slightly larger for random crop
            T.RandomCrop(224),
            T.RandomHorizontalFlip(0.5),
            T.RandomRotation(15),
            T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.1),
            T.RandomGrayscale(0.05),
            T.ToTensor(),
            T.Normalize(IMG_MEAN, IMG_STD),
        ])
    else:  # val / test
        return T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(IMG_MEAN, IMG_STD),
        ])


# ─────────────────────────────────────────────────────────────────────────────
# FILE COLLECTION HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _label_from_folder(folder_name: str) -> Optional[int]:
    """
    Infer 0/1 label from folder name.
    Returns None if folder name is unrecognised (will be skipped).
    """
    name = folder_name.lower().strip()
    if name in POSITIVE_NAMES:
        return 1
    if name in NEGATIVE_NAMES:
        return 0
    # Partial-match fallback
    for kw in POSITIVE_NAMES:
        if kw in name:
            return 1
    for kw in NEGATIVE_NAMES:
        if kw in name:
            return 0
    return None   # unknown folder → skip


def collect_image_samples(root: str) -> List[Tuple[str, int]]:
    """
    Walk <root>/<class_folder>/**/*.{jpg,png,...}
    Returns [(path, label), ...] sorted deterministically.

    FIX OVER ORIGINAL:
      - Graceful handling of missing path
      - Partial-match label inference (handles 'parkinsons_data' etc.)
      - Skips unrecognised folders with a warning instead of crashing
      - Validates file exists before adding
    """
    samples = []
    root_path = Path(root)

    if not root_path.exists():
        print(f"  [WARN] Dataset path not found: {root}")
        return samples

    class_dirs = sorted([d for d in root_path.iterdir() if d.is_dir()])
    if not class_dirs:
        print(f"  [WARN] No subdirectories found in: {root}")
        print(f"         Expected: <root>/healthy/ and <root>/parkinson/")
        return samples

    for class_dir in class_dirs:
        label = _label_from_folder(class_dir.name)
        if label is None:
            print(f"  [WARN] Unrecognised class folder skipped: '{class_dir.name}'")
            print(f"         Rename to something like 'parkinson' or 'healthy'")
            continue

        count_before = len(samples)
        for f in sorted(class_dir.rglob("*")):
            if f.suffix.lower() in IMAGE_EXTS and f.is_file():
                samples.append((str(f), label))
        added = len(samples) - count_before
        tag   = "Parkinson's" if label == 1 else "Healthy"
        print(f"    [{tag}]  {class_dir.name}/  →  {added} images")

    return samples


def collect_video_samples(root: str) -> List[Tuple[str, int]]:
    """
    Same as collect_image_samples but for video files.
    """
    samples = []
    root_path = Path(root)

    if not root_path.exists():
        print(f"  [WARN] Video dataset path not found: {root}")
        return samples

    class_dirs = sorted([d for d in root_path.iterdir() if d.is_dir()])
    if not class_dirs:
        print(f"  [WARN] No subdirectories found in: {root}")
        return samples

    for class_dir in class_dirs:
        label = _label_from_folder(class_dir.name)
        if label is None:
            print(f"  [WARN] Unrecognised folder skipped: '{class_dir.name}'")
            continue

        count_before = len(samples)
        for f in sorted(class_dir.rglob("*")):
            if f.suffix.lower() in VIDEO_EXTS and f.is_file():
                samples.append((str(f), label))
        added = len(samples) - count_before
        tag   = "Parkinson's" if label == 1 else "Healthy"
        print(f"    [{tag}]  {class_dir.name}/  →  {added} videos")

    return samples


# ─────────────────────────────────────────────────────────────────────────────
# INPUT VALIDATION  (used by app.py for uploaded files)
# ─────────────────────────────────────────────────────────────────────────────
def validate_image_file(file_path: str) -> Tuple[bool, str]:
    """
    Returns (is_valid, error_message).
    Checks:
      1. File extension is in IMAGE_EXTS
      2. File magic bytes confirm it's a real image
      3. PIL can open it
    """
    path = Path(file_path)

    # 1. Extension check
    if path.suffix.lower() not in IMAGE_EXTS:
        return False, f"Invalid file type: '{path.suffix}'. Expected image (.jpg, .png, .bmp)"

    # 2. Magic bytes check (imghdr reads first few bytes, doesn't trust extension)
    try:
        detected = imghdr.what(file_path)
        if detected is None:
            return False, "Invalid file type: file does not appear to be a valid image"
    except Exception:
        return False, "File could not be processed: unable to read file header"

    # 3. PIL open check
    try:
        with Image.open(file_path) as img:
            img.verify()   # checks integrity
    except UnidentifiedImageError:
        return False, "Invalid file type: unrecognised image format"
    except Exception as e:
        return False, f"File could not be processed: {str(e)}"

    return True, ""


def validate_video_file(file_path: str) -> Tuple[bool, str]:
    """
    Returns (is_valid, error_message).
    """
    path = Path(file_path)

    if path.suffix.lower() not in VIDEO_EXTS:
        return False, f"Invalid file type: '{path.suffix}'. Expected video (.mp4, .avi, .mov)"

    try:
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            cap.release()
            return False, "File could not be processed: video could not be opened"
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if frame_count == 0:
            return False, "File could not be processed: video has zero frames"
    except Exception as e:
        return False, f"File could not be processed: {str(e)}"

    return True, ""


def is_irrelevant_image(img_array: np.ndarray,
                        min_variance: float = 0.001) -> bool:
    """
    Heuristic: if image is nearly uniform (solid colour, blank, etc.) reject it.
    min_variance: threshold on normalised pixel variance (0–1 scale).
    """
    arr = img_array.astype(np.float32) / 255.0
    return float(arr.var()) < min_variance


# ─────────────────────────────────────────────────────────────────────────────
# DATASETS
# ─────────────────────────────────────────────────────────────────────────────
class ImageDataset(Dataset):
    """
    For MRI and Spiral datasets.
    Handles corrupt files by returning a zero tensor (never crashes training).
    """

    def __init__(self, samples: List[Tuple[str, int]], transform=None):
        self.samples   = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            # Corrupt / unreadable file → silent fallback
            img = Image.new("RGB", (224, 224), color=0)

        if self.transform:
            try:
                img = self.transform(img)
            except Exception:
                img = torch.zeros(3, 224, 224)

        return img, torch.tensor(label, dtype=torch.float32)


class VideoDataset(Dataset):
    """
    For the video tremor dataset.
    Uniformly samples `num_frames` frames from each video.
    Handles corrupt / missing videos by returning zero tensors.
    """

    def __init__(self, samples: List[Tuple[str, int]],
                 num_frames: int = 30, img_size: int = 112):
        self.samples    = samples
        self.num_frames = num_frames
        self.img_size   = img_size
        self._mean      = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self._std       = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.samples)

    def _load_video(self, path: str) -> torch.Tensor:
        cap   = cv2.VideoCapture(path)
        total = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        # Choose frame indices uniformly
        indices = set(np.linspace(0, total - 1, self.num_frames, dtype=int).tolist())
        frames, idx = [], 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if idx in indices:
                frame = cv2.resize(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    (self.img_size, self.img_size)
                )
                frame = (frame.astype(np.float32) / 255.0 - self._mean) / self._std
                frames.append(frame)
                if len(frames) == self.num_frames:
                    break
            idx += 1
        cap.release()

        # Pad if not enough frames
        blank = np.zeros((self.img_size, self.img_size, 3), dtype=np.float32)
        while len(frames) < self.num_frames:
            frames.append(blank)

        arr = np.stack(frames[:self.num_frames])         # (T, H, W, 3)
        return torch.tensor(arr.transpose(0, 3, 1, 2))  # (T, 3, H, W)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            video = self._load_video(path)
        except Exception:
            video = torch.zeros(self.num_frames, 3, self.img_size, self.img_size)

        return video, torch.tensor(label, dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────────────────
# DATASET SPLITTING
# ─────────────────────────────────────────────────────────────────────────────
def split_dataset(dataset: Dataset,
                  val_frac:  float = 0.15,
                  test_frac: float = 0.15,
                  seed: int  = 42):
    """
    Returns (train_subset, val_subset, test_subset).
    """
    n       = len(dataset)
    n_val   = max(1, int(n * val_frac))
    n_test  = max(1, int(n * test_frac))
    n_train = n - n_val - n_test
    if n_train < 1:
        raise ValueError(f"Dataset too small ({n} samples) to split into train/val/test")
    return random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(seed)
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLASS-WEIGHT HELPER  (for imbalanced datasets)
# ─────────────────────────────────────────────────────────────────────────────
def compute_pos_weight(samples: List[Tuple[str, int]]) -> torch.Tensor:
    """
    Returns pos_weight for BCEWithLogitsLoss.
    Formula: n_negative / n_positive
    """
    labels = [s[1] for s in samples]
    n_pos  = sum(labels)
    n_neg  = len(labels) - n_pos
    if n_pos == 0:
        print("  [WARN] No positive samples found — pos_weight set to 1.0")
        return torch.tensor(1.0)
    pw = n_neg / n_pos
    print(f"  Class balance → Healthy: {n_neg}  Parkinson's: {n_pos}  pos_weight: {pw:.2f}")
    return torch.tensor(pw, dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────────────────
# DATALOADER BUILDER  (convenience wrapper)
# ─────────────────────────────────────────────────────────────────────────────
def build_loaders(dataset_type: str,
                  samples: List[Tuple[str, int]],
                  batch_size: int    = 16,
                  num_frames: int    = 30,
                  num_workers: int   = 0) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    dataset_type: "image" | "video"
    Returns (train_loader, val_loader, test_loader)

    NOTE: num_workers defaults to 0 (safe for Windows + PyTorch multiprocessing).
          Set higher on Linux for faster loading.
    """
    if dataset_type == "image":
        ds_train = ImageDataset(samples, transform=get_transforms("train"))
        ds_val   = ImageDataset(samples, transform=get_transforms("val"))
        ds_test  = ImageDataset(samples, transform=get_transforms("test"))
    elif dataset_type == "video":
        ds_train = VideoDataset(samples, num_frames=num_frames)
        ds_val   = VideoDataset(samples, num_frames=num_frames)
        ds_test  = VideoDataset(samples, num_frames=num_frames)
    else:
        raise ValueError(f"dataset_type must be 'image' or 'video', got '{dataset_type}'")

    train_sub, val_sub, test_sub = split_dataset(ds_train)

    # Re-apply correct transforms to val/test subsets
    # (random_split wraps dataset; we pass the right dataset object directly)
    train_loader = DataLoader(
        train_sub, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False
    )
    val_loader = DataLoader(
        ImageDataset([ds_val.samples[i] for i in val_sub.indices],
                     transform=get_transforms("val"))
        if dataset_type == "image" else
        VideoDataset([ds_val.samples[i] for i in val_sub.indices], num_frames=num_frames),
        batch_size=batch_size if dataset_type == "image" else max(1, batch_size // 4),
        shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        ImageDataset([ds_test.samples[i] for i in test_sub.indices],
                     transform=get_transforms("test"))
        if dataset_type == "image" else
        VideoDataset([ds_test.samples[i] for i in test_sub.indices], num_frames=num_frames),
        batch_size=batch_size if dataset_type == "image" else max(1, batch_size // 4),
        shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────────────────────────────────────
# CLI test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Validate paths exist (update to your actual paths before running)
    paths = {
        "MRI"    : r"C:\Users\LENOVO\Downloads\archive (5)\parkinsons_dataset",
        "Spiral" : r"C:\Users\LENOVO\Downloads\archive (6)\spiral",
        "Video"  : r"C:\Users\LENOVO\Downloads\archive (7)",
    }
    for name, p in paths.items():
        exists = Path(p).exists()
        print(f"  {name:8s}  {'✓ Found' if exists else '✗ NOT FOUND'}  {p}")