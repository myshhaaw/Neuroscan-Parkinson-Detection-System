"""
models.py — All PyTorch model architectures for NeuroScan AI
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models


# ──────────────────────────────────────────────────────────────────────────────
# 1. MRI Model  (EfficientNet-B0 backbone + attention)
# ──────────────────────────────────────────────────────────────────────────────

class MRIModel(nn.Module):
    """EfficientNet-B0 fine-tuned for brain MRI feature extraction."""

    def __init__(self, feature_dim: int = 256, pretrained: bool = True):
        super().__init__()
        weights = tv_models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = tv_models.efficientnet_b0(weights=weights)

        # Replace classifier head with feature projector
        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()
        self.backbone = backbone

        self.projector = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):           # x: (B, 3, 224, 224)
        feats = self.backbone(x)    # (B, 1280)
        return self.projector(feats)  # (B, feature_dim)


# ──────────────────────────────────────────────────────────────────────────────
# 2. Video / Tremor Model  (CNN frame encoder + LSTM temporal model)
# ──────────────────────────────────────────────────────────────────────────────

class VideoTremorModel(nn.Module):
    """
    Per-frame CNN encoder (MobileNetV2) + bidirectional LSTM.
    Input:  (B, T, 3, 112, 112) — B=batch, T=frames
    Output: (B, feature_dim)
    """

    def __init__(self, feature_dim: int = 256, num_frames: int = 30,
                 lstm_hidden: int = 128, pretrained: bool = True):
        super().__init__()
        self.num_frames = num_frames

        # Lightweight CNN per-frame encoder
        weights = tv_models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        mobilenet = tv_models.mobilenet_v2(weights=weights)
        # Output after avg-pool: 1280 features
        self.cnn = nn.Sequential(*list(mobilenet.children())[:-1],
                                  nn.AdaptiveAvgPool2d(1),
                                  nn.Flatten())  # → (B*T, 1280)

        # Bidirectional LSTM over time axis
        self.lstm = nn.LSTM(input_size=1280, hidden_size=lstm_hidden,
                            num_layers=2, batch_first=True,
                            bidirectional=True, dropout=0.3)

        self.projector = nn.Sequential(
            nn.Linear(lstm_hidden * 2, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
        )

    def forward(self, x):
        # x: (B, T, 3, H, W)
        B, T, C, H, W = x.shape
        x_flat = x.view(B * T, C, H, W)            # (B*T, 3, H, W)
        frame_feats = self.cnn(x_flat)              # (B*T, 1280)
        frame_feats = frame_feats.view(B, T, -1)    # (B, T, 1280)
        lstm_out, _ = self.lstm(frame_feats)        # (B, T, 256)
        # Global average over time
        pooled = lstm_out.mean(dim=1)               # (B, 256)
        return self.projector(pooled)               # (B, feature_dim)


# ──────────────────────────────────────────────────────────────────────────────
# 3. Scribble / Handwriting Model  (EfficientNet-B0 variant)
# ──────────────────────────────────────────────────────────────────────────────

class ScribbleModel(nn.Module):
    """
    Lightweight CNN for spiral / handwriting irregularity detection.
    Uses channel attention (Squeeze-and-Excitation style).
    """

    def __init__(self, feature_dim: int = 256, pretrained: bool = True):
        super().__init__()
        weights = tv_models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = tv_models.efficientnet_b0(weights=weights)
        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()
        self.backbone = backbone

        self.se = nn.Sequential(
            nn.Linear(in_features, in_features // 4),
            nn.ReLU(inplace=True),
            nn.Linear(in_features // 4, in_features),
            nn.Sigmoid()
        )

        self.projector = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        feats = self.backbone(x)            # (B, 1280)
        gate = self.se(feats)               # (B, 1280)
        feats = feats * gate                # channel attention
        return self.projector(feats)        # (B, feature_dim)


# ──────────────────────────────────────────────────────────────────────────────
# 4. Multi-Modal Fusion Head
# ──────────────────────────────────────────────────────────────────────────────

class FusionClassifier(nn.Module):
    """
    Concatenates [MRI | Video | Scribble] feature vectors and classifies.
    Output: raw logit (apply sigmoid for probability).
    """

    def __init__(self, mri_dim=256, video_dim=256, scribble_dim=256,
                 hidden_dim=512, dropout=0.5):
        super().__init__()
        in_dim = mri_dim + video_dim + scribble_dim  # 768

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 2, 1)   # binary logit
        )

    def forward(self, mri_feat, video_feat, scribble_feat):
        x = torch.cat([mri_feat, video_feat, scribble_feat], dim=1)
        return self.net(x).squeeze(1)


# ──────────────────────────────────────────────────────────────────────────────
# 5. Full End-to-End Model (for convenience during training)
# ──────────────────────────────────────────────────────────────────────────────

class NeuroScanModel(nn.Module):
    """Wraps all sub-models for end-to-end training."""

    def __init__(self, feature_dim=256, pretrained=True):
        super().__init__()
        self.mri_model      = MRIModel(feature_dim, pretrained)
        self.video_model    = VideoTremorModel(feature_dim, pretrained=pretrained)
        self.scribble_model = ScribbleModel(feature_dim, pretrained)
        self.fusion         = FusionClassifier(feature_dim, feature_dim, feature_dim)

    def forward(self, mri, video, scribble):
        mri_feat      = self.mri_model(mri)
        video_feat    = self.video_model(video)
        scribble_feat = self.scribble_model(scribble)
        logit         = self.fusion(mri_feat, video_feat, scribble_feat)
        return logit, (mri_feat, video_feat, scribble_feat)

    def predict(self, mri, video, scribble, threshold=0.5):
        """Returns (label, probability)."""
        self.eval()
        with torch.no_grad():
            logit, _ = self.forward(mri, video, scribble)
            prob = torch.sigmoid(logit)
        label = "Parkinson's Detected" if prob.item() >= threshold else "No Parkinson's Detected"
        return label, prob.item()


if __name__ == "__main__":
    # Quick sanity check
    mri_batch  = torch.randn(2, 3, 224, 224)
    vid_batch  = torch.randn(2, 30, 3, 112, 112)
    scrib_batch= torch.randn(2, 3, 224, 224)

    model = NeuroScanModel(feature_dim=256, pretrained=False)
    logit, feats = model(mri_batch, vid_batch, scrib_batch)
    print(f"Logit shape    : {logit.shape}")   # (2,)
    print(f"MRI feat shape : {feats[0].shape}") # (2, 256)
    print("✓ All models OK")
