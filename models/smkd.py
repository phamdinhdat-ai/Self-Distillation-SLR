"""
Self-Mutual Knowledge Distillation (SMKD) for Continuous Sign Language Recognition
Re-implementation based on: "Self-Mutual Distillation Learning for Continuous Sign Language Recognition"
Hao et al., ICCV 2021
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


# ---------------------------------------------------------------------------
# Visual Module: 2D-ResNet18 + 1D-CNN
# ---------------------------------------------------------------------------

class VisualModule(nn.Module):
    """
    Encodes spatial and short-term temporal information.
    Architecture: 2D-ResNet18 (frame-wise) + 1D-CNN (temporal).
    """

    def __init__(self, d_model: int = 512):
        super().__init__()

        # 2D spatial backbone (shared across all frames)
        backbone = resnet18(weights=None)   # pass weights=ResNet18_Weights.DEFAULT for pretrained
        # Remove the final FC layer; keep average pool → 512-dim feature
        self.spatial_encoder = nn.Sequential(*list(backbone.children())[:-1])  # (B, 512, 1, 1)

        # 1D temporal CNN: C5-P2-C5  (kernel=5, pool-stride=2, kernel=5)
        self.temporal_cnn = nn.Sequential(
            nn.Conv1d(512, d_model, kernel_size=5, padding=2),
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),           # downsample ×2
            nn.Conv1d(d_model, d_model, kernel_size=5, padding=2),
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C, H, W)  – batch of video clips
        Returns:
            lvf: (B, T', d_model)  – Local Visual Features
        """
        B, T, C, H, W = x.shape
        # Frame-wise 2D encoding
        x_flat = x.view(B * T, C, H, W)
        feat = self.spatial_encoder(x_flat)          # (B*T, 512, 1, 1)
        feat = feat.view(B, T, -1)                   # (B, T, 512)

        # Temporal 1D-CNN expects (B, C, T)
        feat = feat.permute(0, 2, 1)                 # (B, 512, T)
        lvf = self.temporal_cnn(feat)                # (B, d_model, T')
        lvf = lvf.permute(0, 2, 1)                   # (B, T', d_model)
        return lvf


# ---------------------------------------------------------------------------
# Contextual Module: 2-layer BiLSTM
# ---------------------------------------------------------------------------

class ContextualModule(nn.Module):
    """
    Encodes long-term context from the visual features via a 2-layer BiLSTM.
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
        # Project back to d_model
        self.proj = nn.Sequential(
            nn.Linear(hidden_size * 2, d_model),
            nn.ReLU(inplace=True),
        )

    def forward(self, lvf: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lvf: (B, T', d_model)
        Returns:
            gcf: (B, T', d_model)  – Global Contextual Features
        """
        out, _ = self.bilstm(lvf)     # (B, T', 2*hidden)
        gcf = self.proj(out)           # (B, T', d_model)
        return gcf


# ---------------------------------------------------------------------------
# Shared Classifier with L2-normalised weights (A-softmax style)
# ---------------------------------------------------------------------------

class NormalisedClassifier(nn.Module):
    """
    Linear classifier whose weight vectors are L2-normalised (no bias).
    This lets cosine similarity drive feature alignment.
    """

    def __init__(self, d_model: int, num_classes: int):
        super().__init__()
        # +1 for CTC blank token
        self.weight = nn.Parameter(torch.randn(num_classes + 1, d_model))
        nn.init.kaiming_uniform_(self.weight)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat: (B, T', d_model)
        Returns:
            logits: (B, T', num_classes+1)
        """
        w = F.normalize(self.weight, p=2, dim=1)          # (C+1, d)
        feat_n = F.normalize(feat, p=2, dim=-1)            # (B, T', d)
        logits = feat_n @ w.t()                            # (B, T', C+1)
        return logits


# ---------------------------------------------------------------------------
# Full SMKD Network
# ---------------------------------------------------------------------------

class SMKD(nn.Module):
    """
    Self-Mutual Knowledge Distillation network.

    Training proceeds in three stages (controlled externally):
      Stage 1 – Synchronous training: shared classifier, CTC on both modules.
      Stage 2 – Gloss segmentation: add CE-LS loss on visual module.
      Stage 3 – Decouple training: independent classifiers, CTC only on contextual.
    """

    def __init__(self, num_classes: int, d_model: int = 512, hidden_size: int = 512):
        super().__init__()
        self.num_classes = num_classes
        self.d_model = d_model

        self.visual_module = VisualModule(d_model=d_model)
        self.contextual_module = ContextualModule(d_model=d_model, hidden_size=hidden_size)

        # Shared classifier (used in stages 1 & 2)
        self.shared_classifier = NormalisedClassifier(d_model, num_classes)

        # Independent classifiers (stage 3)
        self.visual_classifier = NormalisedClassifier(d_model, num_classes)
        self.contextual_classifier = NormalisedClassifier(d_model, num_classes)

        # Training stage: 1, 2, or 3
        self.stage = 1

    def set_stage(self, stage: int):
        assert stage in (1, 2, 3), "Stage must be 1, 2, or 3."
        self.stage = stage

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, T, C, H, W)
        Returns:
            dict with keys: 'logits_v', 'logits_g', 'lvf', 'gcf'
        """
        lvf = self.visual_module(x)           # (B, T', d)
        gcf = self.contextual_module(lvf)     # (B, T', d)

        if self.stage in (1, 2):
            logits_v = self.shared_classifier(lvf)   # (B, T', C+1)
            logits_g = self.shared_classifier(gcf)
        else:
            # Stage 3: decouple
            logits_v = self.visual_classifier(lvf)
            logits_g = self.contextual_classifier(gcf)

        return {
            "logits_v": logits_v,
            "logits_g": logits_g,
            "lvf": lvf,
            "gcf": gcf,
        }

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Greedy CTC decoding using contextual module output."""
        with torch.no_grad():
            out = self.forward(x)
            log_probs = F.log_softmax(out["logits_g"], dim=-1)
            # (B, T', C+1) → argmax → remove blanks and repeats
            pred = log_probs.argmax(dim=-1)            # (B, T')
        return pred
