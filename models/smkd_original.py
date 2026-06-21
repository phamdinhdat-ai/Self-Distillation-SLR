"""
Original Self-Mutual Knowledge Distillation (SMKD) for Continuous SLR
Paper: "Self-Mutual Distillation Learning for Continuous Sign Language Recognition"
Hao et al., ICCV 2021

This is the faithful reproduction of the paper's architecture:
  - ResNet-18 spatial backbone (frame-wise)
  - Standard Conv1d temporal sub-network (2 layers)
  - 2-layer BiLSTM contextual module
  - L2-normalised shared/independent classifiers
  - Three-stage training: sync → GSBA → decouple
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


# ---------------------------------------------------------------------------
# Visual Module: ResNet-18 + 2-layer Conv1d (paper: Section 3.2)
# ---------------------------------------------------------------------------

class VisualModule(nn.Module):
    """
    Encodes spatial and short-term temporal information.

    Architecture (paper):
      - ResNet-18 (without final FC + pooling) → 512-dim frame features
      - Conv1d(512, 512, 5) → BN → ReLU → MaxPool1d(2)
      - Conv1d(512, 512, 5) → BN → ReLU
      - Output: (B, T/2, 512) Local Visual Features (LVF)
    """

    def __init__(self, d_model: int = 512):
        super().__init__()
        # ResNet-18 backbone: drop final FC and avgpool
        backbone = resnet18(weights=None)
        self.spatial_encoder = nn.Sequential(*list(backbone.children())[:-1])
        self.backbone_dim = 512

        # Temporal 1D-CNN (mapped to d_model)
        self.temporal_cnn = nn.Sequential(
            nn.Conv1d(512, d_model, kernel_size=5, padding=2),
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(d_model, d_model, kernel_size=5, padding=2),
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C, H, W)
        Returns:
            lvf: (B, T', d_model)  -- Local Visual Features
        """
        B, T, C, H, W = x.shape
        x_flat = x.view(B * T, C, H, W)

        # Frame-wise 2D encoding
        feat = self.spatial_encoder(x_flat)           # (B*T, 512, 1, 1)
        feat = feat.view(B, T, self.backbone_dim)     # (B, T, 512)

        # Temporal 1D-CNN: (B, C, T)
        feat_t = feat.permute(0, 2, 1)                # (B, 512, T)
        lvf_t = self.temporal_cnn(feat_t)             # (B, d_model, T')
        lvf = lvf_t.permute(0, 2, 1)                   # (B, T', d_model)
        return lvf


# ---------------------------------------------------------------------------
# Contextual Module: 2-layer BiLSTM (paper: Section 3.3)
# ---------------------------------------------------------------------------

class ContextualModule(nn.Module):
    """
    Encodes long-range context from visual features.

    Architecture (paper):
      - 2-layer BiLSTM(hidden_size=512)
      - Linear(1024 → d_model) + ReLU projection
      - Output: (B, T', d_model) Global Contextual Features (GCF)
    """

    def __init__(self, d_model: int = 512, hidden_size: int = 512):
        super().__init__()
        self.bilstm = nn.LSTM(
            input_size=d_model,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )
        self.proj = nn.Sequential(
            nn.Linear(hidden_size * 2, d_model),
            nn.ReLU(inplace=True),
        )

    def forward(self, lvf: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lvf: (B, T', d_model)
        Returns:
            gcf: (B, T', d_model) -- Global Contextual Features
        """
        out, _ = self.bilstm(lvf)     # (B, T', 2*hidden_size)
        gcf = self.proj(out)           # (B, T', d_model)
        return gcf


# ---------------------------------------------------------------------------
# Normalised Classifier (paper: Section 3.4 – A-softmax style)
# ---------------------------------------------------------------------------

class NormalisedClassifier(nn.Module):
    """
    Linear classifier with L2-normalised weight vectors (no bias).

    Cosine similarity bounded in [-1, 1] makes softmax nearly uniform
    at initialization. A temperature scale (τ) sharpens the distribution
    so CTC can form strong peaks. Paper uses τ = 20.
    """

    def __init__(self, d_model: int, num_classes: int, scale: float = 20.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_classes + 1, d_model))
        nn.init.kaiming_uniform_(self.weight)
        self.scale = scale

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat: (B, T', d_model)
        Returns:
            logits: (B, T', num_classes + 1)
        """
        w = F.normalize(self.weight, p=2, dim=1)
        feat_n = F.normalize(feat, p=2, dim=-1)
        logits = feat_n @ w.t() * self.scale
        return logits


# ---------------------------------------------------------------------------
# Full SMKD Network (paper: Section 3)
# ---------------------------------------------------------------------------

class SMKD(nn.Module):
    """
    Self-Mutual Knowledge Distillation network.

    Three training stages:
      1. Synchronous training: shared classifier, CTC on both branches.
      2. Gloss segmentation: shared classifier, CTC + CE-LS on visual.
      3. Decouple: independent classifiers, CTC on visual freezes, contextual trains.
    """

    def __init__(
        self,
        num_classes: int,
        d_model: int = 512,
        hidden_size: int = 512,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.d_model = d_model

        self.visual_module = VisualModule(d_model=d_model)
        self.contextual_module = ContextualModule(d_model=d_model, hidden_size=hidden_size)

        # Shared classifier (stages 1 and 2)
        self.shared_classifier = NormalisedClassifier(d_model, num_classes)

        # Independent classifiers (stage 3)
        self.visual_classifier = NormalisedClassifier(d_model, num_classes)
        self.contextual_classifier = NormalisedClassifier(d_model, num_classes)

        self.stage = 1

    def set_stage(self, stage: int):
        assert stage in (1, 2, 3), "Stage must be 1, 2, or 3."
        self.stage = stage

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, T, C, H, W)
        Returns:
            dict with: 'logits_v', 'logits_g', 'lvf', 'gcf'
        """
        lvf = self.visual_module(x)
        gcf = self.contextual_module(lvf)

        if self.stage in (1, 2):
            logits_v = self.shared_classifier(lvf)
            logits_g = self.shared_classifier(gcf)
        else:
            logits_v = self.visual_classifier(lvf)
            logits_g = self.contextual_classifier(gcf)

        return {
            "logits_v": logits_v,
            "logits_g": logits_g,
            "lvf": lvf,
            "gcf": gcf,
        }

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Greedy CTC decoding from contextual module."""
        with torch.no_grad():
            out = self.forward(x)
            log_probs = F.log_softmax(out["logits_g"], dim=-1)
            pred = log_probs.argmax(dim=-1)
        return pred
