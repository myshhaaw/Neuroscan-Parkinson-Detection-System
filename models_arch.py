"""
models_arch.py — NeuroScan AI v3.0
────────────────────────────────────────────────────────────────────────────────
Multi-modal, multi-backbone architecture with GradCAM support.

Models per modality:
  • CustomCNN    — lightweight 5-block CNN (baseline, no pretrained weights)
  • MobileNetV2  — pretrained, fine-tuned
  • EfficientNetB0 — pretrained, fine-tuned with SE attention

GradCAM hooks are registered on the final conv layer of each backbone so
inference can produce spatial heatmaps without re-running the model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from typing import Optional, Dict, Tuple
import numpy as np
from segmentation import UNet, unsupervised_roi_mask

# ──────────────────────────────────────────────────────────────────────────────
# GRAD-CAM UTILITY
# ──────────────────────────────────────────────────────────────────────────────
FEATURE_DIM = 256

class GradCAM:
    """
    Computes class-activation maps for any model by hooking the target layer.

    Usage:
        cam = GradCAM(model, target_layer=model.backbone.features[-1])
        heatmap = cam(input_tensor)   # numpy (H, W) in [0, 1]
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _, __, output):
        self.activations = output.detach()

    def _save_gradient(self, _, __, grad_output):
        self.gradients = grad_output[0].detach()

    def remove(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()

    def __call__(self, x: torch.Tensor) -> np.ndarray:
        """
        x: (1, C, H, W) — single image tensor on the same device as model.
        Returns: numpy float32 array (H, W) normalised to [0, 1].
        """
        self.model.eval()
        x = x.requires_grad_(True)

        logit = self.model(x)                         # forward
        self.model.zero_grad()
        logit.backward(torch.ones_like(logit))         # backward w.r.t. logit

        # GAP over spatial dims
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)

        # Resize to input spatial dims
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalise
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-6:
            cam = (cam - cam_min) / (cam_max - cam_min)
        return cam.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Custom CNN  (baseline — no pretrained weights)
# ──────────────────────────────────────────────────────────────────────────────

class _ConvBnRelu(nn.Sequential):
    def __init__(self, in_c, out_c, k=3, s=1, p=1):
        super().__init__(
            nn.Conv2d(in_c, out_c, k, s, p, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )


class CustomCNNEncoder(nn.Module):
    """
    5-block custom CNN designed to avoid overfitting via:
      • BatchNorm after every conv
      • SpatialDropout2d between blocks
      • Global-Average-Pool (no FC bottleneck on spatial dims)
    """
    def __init__(self, feature_dim: int = 256, in_channels: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1  224→112
            _ConvBnRelu(in_channels, 32), _ConvBnRelu(32, 32),
            nn.MaxPool2d(2), nn.Dropout2d(0.05),
            # Block 2  112→56
            _ConvBnRelu(32, 64), _ConvBnRelu(64, 64),
            nn.MaxPool2d(2), nn.Dropout2d(0.05),
            # Block 3  56→28
            _ConvBnRelu(64, 128), _ConvBnRelu(128, 128),
            nn.MaxPool2d(2), nn.Dropout2d(0.1),
            # Block 4  28→14
            _ConvBnRelu(128, 256), _ConvBnRelu(256, 256),
            nn.MaxPool2d(2), nn.Dropout2d(0.1),
            # Block 5  14→7  ← GradCAM hooks here
            _ConvBnRelu(256, 512), _ConvBnRelu(512, 512),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
        )
        # expose last conv layer for GradCAM
        self.gradcam_layer = self.features[-1]

    def forward(self, x):               # x: (B, 3, 224, 224)
        f = self.features(x)            # (B, 512, 7, 7)
        return self.projector(self.gap(f))   # (B, feature_dim)


# ──────────────────────────────────────────────────────────────────────────────
# 2. MobileNetV2  (lightweight pretrained backbone)
# ──────────────────────────────────────────────────────────────────────────────

class MobileNetV2Encoder(nn.Module):
    def __init__(self, feature_dim: int = 256, pretrained: bool = True):
        super().__init__()
        weights = tv_models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        backbone = tv_models.mobilenet_v2(weights=weights)
        # strip the classifier
        self.features = backbone.features          # (B, 1280, 7, 7)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1280, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
        )
        # GradCAM target: last conv block
        self.gradcam_layer = self.features[-1]

    def forward(self, x):
        f = self.features(x)
        return self.projector(self.gap(f))


# ──────────────────────────────────────────────────────────────────────────────
# 3. EfficientNetB0  (pretrained + Squeeze-and-Excitation style gate)
# ──────────────────────────────────────────────────────────────────────────────

class EfficientNetB0Encoder(nn.Module):
    def __init__(self, feature_dim: int = 256, pretrained: bool = True):
        super().__init__()
        weights = tv_models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = tv_models.efficientnet_b0(weights=weights)
        in_features = backbone.classifier[1].in_features  # 1280
        backbone.classifier = nn.Identity()
        self.backbone = backbone                           # includes avgpool

        # channel attention gate (SE)
        self.se = nn.Sequential(
            nn.Linear(in_features, in_features // 8),
            nn.ReLU(inplace=True),
            nn.Linear(in_features // 8, in_features),
            nn.Sigmoid(),
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
        # GradCAM target: last features block
        self.gradcam_layer = self.backbone.features[-1]

    def forward(self, x):
        f = self.backbone(x)               # (B, 1280)
        f = f * self.se(f)                 # channel-wise attention
        return self.projector(f)           # (B, feature_dim)


# ──────────────────────────────────────────────────────────────────────────────
# 4. Video / Tremor Model  (CNN frame encoder + bidirectional LSTM)
# ──────────────────────────────────────────────────────────────────────────────

class VideoTremorModel(nn.Module):
    """
    Per-frame CNN encoder chosen from {custom, mobilenet, efficientnet}
    fed into a bidirectional LSTM for temporal modelling.

    Input:  (B, T, 3, H, W)
    Output: (B, feature_dim)
    """
    def __init__(self, backbone: str = "mobilenet",
                 feature_dim: int = 256,
                 num_frames: int = 30,
                 lstm_hidden: int = 128,
                 pretrained: bool = True):
        super().__init__()
        self.num_frames = num_frames
        self.backbone_name = backbone

        if backbone == "mobilenet":
            weights = tv_models.MobileNet_V2_Weights.DEFAULT if pretrained else None
            mv2 = tv_models.mobilenet_v2(weights=weights)
            self.cnn = nn.Sequential(*list(mv2.children())[:-1],
                                     nn.AdaptiveAvgPool2d(1),
                                     nn.Flatten())
            cnn_out = 1280
        elif backbone == "efficientnet":
            weights = tv_models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            eff = tv_models.efficientnet_b0(weights=weights)
            eff.classifier = nn.Identity()
            self.cnn = eff
            cnn_out = 1280
        else:  # custom
            self.cnn = nn.Sequential(
                _ConvBnRelu(3, 32), nn.MaxPool2d(2),
                _ConvBnRelu(32, 64), nn.MaxPool2d(2),
                _ConvBnRelu(64, 128), nn.MaxPool2d(2),
                _ConvBnRelu(128, 256),
                nn.AdaptiveAvgPool2d(1), nn.Flatten()
            )
            cnn_out = 256

        self.lstm = nn.LSTM(
            input_size=cnn_out, hidden_size=lstm_hidden,
            num_layers=2, batch_first=True,
            bidirectional=True, dropout=0.3
        )
        self.projector = nn.Sequential(
            nn.Linear(lstm_hidden * 2, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        x_flat = x.view(B * T, C, H, W)
        frame_feats = self.cnn(x_flat).view(B, T, -1)
        lstm_out, _ = self.lstm(frame_feats)
        pooled = lstm_out.mean(dim=1)
        return self.projector(pooled)


# ──────────────────────────────────────────────────────────────────────────────
# 5. Modality-specific wrapper  (selects backbone per modality)
# ──────────────────────────────────────────────────────────────────────────────

def build_image_encoder(backbone: str, feature_dim: int = 256,
                        pretrained: bool = True) -> nn.Module:
    """
    Factory that returns an image encoder with a `gradcam_layer` attribute.
    backbone ∈ {"custom", "mobilenet", "efficientnet"}
    """
    if backbone == "custom":
        return CustomCNNEncoder(feature_dim)
    elif backbone == "mobilenet":
        return MobileNetV2Encoder(feature_dim, pretrained)
    elif backbone == "efficientnet":
        return EfficientNetB0Encoder(feature_dim, pretrained)
    else:
        raise ValueError(f"Unknown backbone: {backbone}")


# ──────────────────────────────────────────────────────────────────────────────
# 6. Fusion Classifier
# ──────────────────────────────────────────────────────────────────────────────

class FusionClassifier(nn.Module):
    """
    Late-fusion: concat [MRI | Video | Scribble] features → binary logit.
    Uses a 3-layer MLP with GELU, BatchNorm, and progressive dropout.
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
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, mri_feat, video_feat, scribble_feat):
        x = torch.cat([mri_feat, video_feat, scribble_feat], dim=1)
        return self.net(x).squeeze(1)   # raw logit


# ──────────────────────────────────────────────────────────────────────────────
# 7. End-to-end NeuroScanModel
# ──────────────────────────────────────────────────────────────────────────────

class NeuroScanModel(nn.Module):
    """
    Full multi-modal model.

    backbone_mri / backbone_video / backbone_scribble ∈
        {"custom", "mobilenet", "efficientnet"}

    forward() returns (logit, (mri_feat, video_feat, scribble_feat))
    """
    def __init__(self,
                 backbone_mri: str = "efficientnet",
                 backbone_video: str = "mobilenet",
                 backbone_scribble: str = "efficientnet",
                 feature_dim: int = 256,
                 pretrained: bool = True,
                 use_segmentation_supervision: bool = False):
        super().__init__()
        self.mri_model      = build_image_encoder(backbone_mri,      feature_dim, pretrained)
        self.video_model    = VideoTremorModel(backbone_video, feature_dim, pretrained=pretrained)
        self.scribble_model = build_image_encoder(backbone_scribble, feature_dim, pretrained)
        self.seg_model      = UNet()
        self.use_segmentation_supervision = use_segmentation_supervision
        self.fusion         = FusionClassifier(feature_dim, feature_dim, feature_dim)

    def _segment_mri(self, mri: torch.Tensor) -> torch.Tensor:
        if self.use_segmentation_supervision:
            return self.seg_model(mri)
        return unsupervised_roi_mask(mri)

    def forward(self, mri, video, scribble):
        mask = self._segment_mri(mri)
        mri = mri * mask
        mf  = self.mri_model(mri)
        vf  = self.video_model(video)
        sf  = self.scribble_model(scribble)
        logit = self.fusion(mf, vf, sf)
        return logit, (mf, vf, sf)

    def predict(self, mri, video, scribble):
        """Returns (probability_float, confidence_float, feature_dict)."""
        self.eval()
        with torch.no_grad():
            logit, (mf, vf, sf) = self.forward(mri, video, scribble)
            prob = torch.sigmoid(logit).item()
        confidence = max(prob, 1.0 - prob)
        return prob, confidence, {"mri": mf, "video": vf, "scribble": sf}
    
# ──────────────────────────────────────────────────────────────────────────────
# 8. get_model (REQUIRED for train_compare.py)
# ──────────────────────────────────────────────────────────────────────────────

def get_model(model_name: str,
              pretrained: bool = True,
              freeze_backbone: bool = True) -> nn.Module:
    """
    Returns a classification model (encoder + binary head).
    Compatible with train_compare.py
    """

    model_name = model_name.lower()

    if model_name == "custom_cnn":
        encoder = CustomCNNEncoder(feature_dim=FEATURE_DIM)

    elif model_name == "mobilenetv2":
        encoder = MobileNetV2Encoder(feature_dim=FEATURE_DIM,
                                     pretrained=pretrained)

    elif model_name == "efficientnetb0":
        encoder = EfficientNetB0Encoder(feature_dim=FEATURE_DIM,
                                        pretrained=pretrained)

    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Freeze backbone if required
    if freeze_backbone:
        for p in encoder.parameters():
            p.requires_grad = False

    # Final classifier head
    class ModelWrapper(nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder
            self.classifier = nn.Linear(FEATURE_DIM, 1)

        def forward(self, x):
            feat = self.encoder(x)
            return self.classifier(feat).squeeze(1)

        def unfreeze_all(self):
            for p in self.encoder.parameters():
                p.requires_grad = True

    return ModelWrapper(encoder)

# ──────────────────────────────────────────────────────────────────────────────
# SANITY CHECK
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    for bb in ("custom", "mobilenet", "efficientnet"):
        enc = build_image_encoder(bb, 256, pretrained=False)
        x   = torch.randn(2, 3, 224, 224)
        out = enc(x)
        print(f"[{bb:12s}] output: {out.shape}  ✓")

    vid = VideoTremorModel("mobilenet", 256, pretrained=False)
    v   = torch.randn(2, 10, 3, 112, 112)
    print(f"[video       ] output: {vid(v).shape}  ✓")

    full = NeuroScanModel(pretrained=False)
    logit, _ = full(
        torch.randn(2, 3, 224, 224),
        torch.randn(2, 10, 3, 112, 112),
        torch.randn(2, 3, 224, 224),
    )
    
    print(f"[fusion logit] output: {logit.shape}  ✓")
    print("All architecture checks passed ✓")
    

