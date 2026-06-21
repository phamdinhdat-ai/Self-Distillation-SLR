"""
Self-Mutual Knowledge Distillation (SMKD) for Continuous Sign Language Recognition
Re-implementation based on: "Self-Mutual Distillation Learning for Continuous Sign Language Recognition"
Hao et al., ICCV 2021

Enhanced with: depthwise-separable temporal convs, multi-scale temporal branches,
lightweight backbone support, and TSM (temporal shift module).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, mobilenet_v3_small, efficientnet_b0


# ---------------------------------------------------------------------------
# Backbone factory
# ---------------------------------------------------------------------------

BACKBONE_OUTPUT_DIMS = {
    "resnet18": 512,
    "mobilenet_v3_small": 576,
    "efficientnet_b0": 1280,
}


def build_backbone(name: str, pretrained: bool = False) -> tuple:
    """
    Returns (spatial_encoder, feature_dim).

    Args:
        name: one of 'resnet18', 'mobilenet_v3_small', 'efficientnet_b0'
        pretrained: if True, load ImageNet-pretrained weights
    """
    from torchvision.models import (
        ResNet18_Weights, MobileNet_V3_Small_Weights,
        EfficientNet_B0_Weights,
    )
    if name == "resnet18":
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)
        return nn.Sequential(*list(backbone.children())[:-1]), 512
    elif name == "mobilenet_v3_small":
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = mobilenet_v3_small(weights=weights)
        modules = list(backbone.children())[:-1]  # drop the classifier head
        return nn.Sequential(*modules, nn.AdaptiveAvgPool2d((1, 1))), 576
    elif name == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = efficientnet_b0(weights=weights)
        modules = list(backbone.children())[:-1]  # drop classifier
        return nn.Sequential(*modules, nn.AdaptiveAvgPool2d((1, 1))), 1280
    else:
        raise ValueError(f"Unknown backbone: {name}. "
                         f"Choose from: {list(BACKBONE_OUTPUT_DIMS.keys())}")


# ---------------------------------------------------------------------------
# Depthwise-separable Conv1d block
# ---------------------------------------------------------------------------

class DepthwiseSeparableConv1d(nn.Module):
    """Depthwise → Pointwise 1D conv, ~8-9x fewer params than standard Conv1d."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int):
        super().__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size,
                                   padding=padding, groups=in_channels)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


# ---------------------------------------------------------------------------
# Temporal Shift Module (TSM) – zero-parameter temporal modeling
# ---------------------------------------------------------------------------

class TemporalShift(nn.Module):
    """
    Shifts part of the channels along the time dimension.
    Fold this into a 2D CNN by processing (B*T, C, H, W) → shift → (B*T, C, H, W).
    """

    def __init__(self, n_segment: int = 8, shift_div: int = 8):
        super().__init__()
        self.n_segment = n_segment
        self.fold_div = shift_div

    def forward(self, x: torch.Tensor, T: int) -> torch.Tensor:
        """
        Args:
            x: (B*T, C, H, W)
            T: number of frames
        Returns:
            x_shifted: (B*T, C, H, W) with channels shifted along time
        """
        B_T, C, H, W = x.shape
        B = B_T // T
        x = x.view(B, T, C, H, W)                     # (B, T, C, H, W)
        fold = C // self.fold_div
        if fold == 0:
            return x.view(B_T, C, H, W)

        x_out = x.clone()
        # Shift 1/8 channels forward, 1/8 channels backward
        x_out[:, :-1, :fold, :, :] = x[:, 1:, :fold, :, :]           # forward shift
        x_out[:, 1:, fold:2*fold, :, :] = x[:, :-1, fold:2*fold, :, :]  # backward shift
        return x_out.view(B_T, C, H, W)


# ---------------------------------------------------------------------------
# Gloss Boundary Detector – auxiliary head on LVF
# ---------------------------------------------------------------------------

class GlossBoundaryDetector(nn.Module):
    """
    Lightweight binary classification head that predicts per-frame gloss
    onset/offset labels derived from GSBA segment boundaries.

    Gives the visual module a structured frame-level objective independent
    of CTC's weak gradient signal.  Only active during stage 2.

    label[t] = 1 if frame t is the first frame of a new GSBA gloss segment
    """

    def __init__(self, d_model: int, hidden_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model // hidden_ratio),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model // hidden_ratio, 2),  # binary: boundary / non-boundary
        )

    def forward(self, lvf: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lvf: (B, T', d_model)
        Returns:
            boundary_logits: (B, T', 2)
        """
        return self.net(lvf)


# ---------------------------------------------------------------------------
# Visual Module: 2D-backbone + 1D-CNN  (enhanced)
# ---------------------------------------------------------------------------

class VisualModule(nn.Module):
    """
    Encodes spatial and short-term temporal information.

    Architecture: 2D backbone (frame-wise) + 1D-CNN (temporal).
    Supports: backbone selection, depthwise-separable convs, multi-scale temporal
    branches, TSM, and optional gloss boundary detector head.
    """

    def __init__(
        self,
        d_model: int = 512,
        backbone_name: str = "resnet18",
        temporal_conv_type: str = "standard",  # "standard" | "depthwise_separable"
        multi_scale: bool = False,
        multi_scale_dilations: list = None,
        use_tsm: bool = False,
        pretrained_backbone: bool = False,
        freeze_backbone: bool = False,
        use_boundary_head: bool = False,
    ):
        super().__init__()

        # 2D spatial backbone
        self.spatial_encoder, bb_dim = build_backbone(backbone_name, pretrained=pretrained_backbone)
        self.backbone_dim = bb_dim
        self.backbone_name = backbone_name
        self.pretrained = pretrained_backbone

        # Optionally freeze the backbone for feature extraction
        if freeze_backbone:
            for p in self.spatial_encoder.parameters():
                p.requires_grad_(False)

        self.use_tsm = use_tsm and backbone_name == "resnet18"
        self.tsm = TemporalShift() if self.use_tsm else None

        # Adapter to map backbone output to d_model if needed
        if bb_dim != d_model and not multi_scale:
            self.backbone_adapter = nn.Linear(bb_dim, d_model)
        else:
            self.backbone_adapter = nn.Identity()

        # 1D temporal CNN
        self.temporal_conv_type = temporal_conv_type
        ConvLayer = DepthwiseSeparableConv1d if temporal_conv_type == "depthwise_separable" else nn.Conv1d

        in_ch = bb_dim
        self.temporal_cnn = nn.Sequential(
            ConvLayer(in_ch, d_model, kernel_size=5, padding=2),
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            ConvLayer(d_model, d_model, kernel_size=5, padding=2),
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True),
        )

        # Multi-scale temporal branch (optional)
        self.multi_scale = multi_scale
        if multi_scale:
            dilations = multi_scale_dilations or [2]
            self.ms_branches = nn.ModuleList()
            for dil in dilations:
                self.ms_branches.append(nn.Sequential(
                    ConvLayer(bb_dim, d_model, kernel_size=5, padding=2 * dil, dilation=dil),
                    nn.BatchNorm1d(d_model),
                    nn.ReLU(inplace=True),
                    nn.MaxPool1d(kernel_size=2, stride=2),
                ))
            # Fusion: concat (main + branches) → project back to d_model
            n_branches = 1 + len(self.ms_branches)
            self.ms_fusion = nn.Sequential(
                nn.Conv1d(d_model * n_branches, d_model, kernel_size=1),
                nn.BatchNorm1d(d_model),
                nn.ReLU(inplace=True),
            )

        # Optional gloss boundary detector (only active in stage 2)
        self.use_boundary_head = use_boundary_head
        self.boundary_head = GlossBoundaryDetector(d_model) if use_boundary_head else None

    def forward(self, x: torch.Tensor, return_boundary: bool = False):
        """
        Args:
            x: (B, T, C, H, W)  – batch of video clips
            return_boundary: if True and boundary head is enabled, also return boundary_logits
        Returns:
            lvf: (B, T', d_model)  – Local Visual Features
            (optionally) boundary_logits: (B, T', 2)
        """
        B, T, C, H, W = x.shape
        x_flat = x.view(B * T, C, H, W)

        # TSM shift (if enabled)
        if self.use_tsm and self.tsm is not None:
            x_flat = self.tsm(x_flat, T)

        # Frame-wise 2D encoding
        feat = self.spatial_encoder(x_flat)          # (B*T, bb_dim, 1, 1)
        feat = feat.view(B, T, self.backbone_dim)    # (B, T, bb_dim)

        # Temporal 1D-CNN expects (B, C, T)
        feat_t = feat.permute(0, 2, 1)               # (B, bb_dim, T)

        if self.multi_scale:
            # Main branch
            main_feat = self.temporal_cnn(feat_t)     # (B, d_model, T')
            # Multi-scale branches
            ms_feats = [branch(feat_t) for branch in self.ms_branches]  # each (B, d_model, T')
            # Concat and fuse
            all_feats = torch.cat([main_feat] + ms_feats, dim=1)  # (B, d_model*n_branches, T')
            lvf_t = self.ms_fusion(all_feats)                      # (B, d_model, T')
        else:
            # temporal_cnn already maps bb_dim → d_model via its first Conv1d
            lvf_t = self.temporal_cnn(feat_t)          # (B, d_model, T')

        lvf = lvf_t.permute(0, 2, 1)                   # (B, T', d_model)

        if return_boundary and self.use_boundary_head and self.boundary_head is not None:
            boundary_logits = self.boundary_head(lvf)   # (B, T', 2)
            return lvf, boundary_logits

        return lvf


# ---------------------------------------------------------------------------
# Conformer Convolution Module
# ---------------------------------------------------------------------------

class ConformerConvModule(nn.Module):
    """
    Depthwise-separable convolution with gating (GLU) and batch norm.
    Sandwiched between the two FFN halves in a Conformer block.
    """

    def __init__(self, d_model: int, kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.pointwise1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        padding = kernel_size // 2
        self.depthwise = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size,
            padding=padding, groups=d_model,
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.activation = nn.SiLU()
        self.pointwise2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.layer_norm(x)
        x = x.permute(0, 2, 1)
        x = self.pointwise1(x)
        x = self.glu(x)
        x = self.depthwise(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pointwise2(x)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)
        return residual + x


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
# Contextual Module: Lightweight Transformer (Conformer-style)
# ---------------------------------------------------------------------------

class TransformerContextualModule(nn.Module):
    """
    Encodes long-term context via a lightweight Transformer encoder.

    Replaces BiLSTM with 2× self-attention blocks + positional encoding.
    Better GPU parallelism than BiLSTM; similar param count (~10.3M vs ~10.5M).
    """

    def __init__(self, d_model: int = 512, num_heads: int = 4, num_layers: int = 2,
                 ff_expansion: int = 2, dropout: float = 0.3, max_len: int = 200):
        super().__init__()
        self.d_model = d_model
        self.pos_encoding = nn.Parameter(
            torch.randn(1, max_len, d_model) * 0.02
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * ff_expansion,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=True,  # Pre-LN for better stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, lvf: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lvf: (B, T', d_model)
        Returns:
            gcf: (B, T', d_model)
        """
        T = lvf.size(1)
        pos = self.pos_encoding[:, :T, :]
        x = lvf + pos
        gcf = self.transformer(x)
        return gcf


# ---------------------------------------------------------------------------
# Contextual Module: Full Conformer (convolution-augmented Transformer)
# ---------------------------------------------------------------------------

class ConformerContextualModule(nn.Module):
    """
    Encodes long-term context via Conformer blocks.

    Each Conformer block = FFN(half) -> MHSA -> Conv -> FFN(half).
    Combines local temporal modelling (depthwise conv) with global
    context (self-attention).  Fully parallelisable across T.

    Ref: Gulati et al., "Conformer: Convolution-augmented Transformer
    for Speech Recognition", Interspeech 2020.
    """

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_expansion: int = 2,
        conv_kernel: int = 31,
        dropout: float = 0.3,
        max_len: int = 200,
    ):
        super().__init__()
        self.d_model = d_model
        self.pos_encoding = nn.Parameter(
            torch.randn(1, max_len, d_model) * 0.02
        )
        self.layers = nn.ModuleList([
            _ConformerBlock(d_model, num_heads, ff_expansion, conv_kernel, dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, lvf: torch.Tensor) -> torch.Tensor:
        T = lvf.size(1)
        pos = self.pos_encoding[:, :T, :]
        x = lvf + pos
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)


class _ConformerBlock(nn.Module):
    """Single Conformer block: FFN(half) -> MHSA -> Conv -> FFN(half)."""

    def __init__(self, d_model, num_heads, ff_expansion, conv_kernel, dropout):
        super().__init__()
        self.ff1 = _FeedForwardModule(d_model, ff_expansion, dropout)
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True,
        )
        self.attn_dropout = nn.Dropout(dropout)
        self.conv_module = ConformerConvModule(d_model, conv_kernel, dropout)
        self.ff2 = _FeedForwardModule(d_model, ff_expansion, dropout)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + 0.5 * self.ff1(x)
        residual = x
        x = self.attn_norm(x)
        x, _ = self.attn(x, x, x)
        x = residual + self.attn_dropout(x)
        x = self.conv_module(x)
        x = x + 0.5 * self.ff2(x)
        return self.final_norm(x)


class _FeedForwardModule(nn.Module):
    """Pre-LN feed-forward with SiLU activation."""

    def __init__(self, d_model, ff_expansion, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_model * ff_expansion)
        self.activation = nn.SiLU()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model * ff_expansion, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return residual + x


# ---------------------------------------------------------------------------
# Shared Classifier with L2-normalised weights (A-softmax style)
# ---------------------------------------------------------------------------

class NormalisedClassifier(nn.Module):
    """
    Linear classifier whose weight vectors are L2-normalised (no bias).
    This lets cosine similarity drive feature alignment.

    Cosine similarities are bounded in [-1, 1], making softmax nearly uniform
    at initialization. A temperature scale sharpens the distribution so CTC can
    form strong peaks (typical values: 16–32).
    """

    def __init__(self, d_model: int, num_classes: int, scale: float = 20.0):
        super().__init__()
        # +1 for CTC blank token
        self.weight = nn.Parameter(torch.randn(num_classes + 1, d_model))
        nn.init.kaiming_uniform_(self.weight)
        self.scale = scale

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat: (B, T', d_model)
        Returns:
            logits: (B, T', num_classes+1)
        """
        w = F.normalize(self.weight, p=2, dim=1)          # (C+1, d)
        feat_n = F.normalize(feat, p=2, dim=-1)            # (B, T', d)
        logits = feat_n @ w.t() * self.scale               # (B, T', C+1)
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

    def __init__(
        self,
        num_classes: int,
        d_model: int = 512,
        hidden_size: int = 512,
        backbone: str = "resnet18",
        temporal_conv_type: str = "standard",
        multi_scale_temporal: bool = False,
        multi_scale_dilation_rates: list = None,
        use_tsm: bool = False,
        pretrained_backbone: bool = False,
        freeze_backbone: bool = False,
        context_type: str = "bilstm",  # "bilstm" | "transformer" | "conformer"
        use_boundary_head: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.d_model = d_model

        self.visual_module = VisualModule(
            d_model=d_model,
            backbone_name=backbone,
            temporal_conv_type=temporal_conv_type,
            multi_scale=multi_scale_temporal,
            multi_scale_dilations=multi_scale_dilation_rates,
            use_tsm=use_tsm,
            pretrained_backbone=pretrained_backbone,
            freeze_backbone=freeze_backbone,
            use_boundary_head=use_boundary_head,
        )

        if context_type == "transformer":
            self.contextual_module = TransformerContextualModule(
                d_model=d_model, num_heads=4, num_layers=2,
                ff_expansion=2, dropout=0.3)
        elif context_type == "conformer":
            self.contextual_module = ConformerContextualModule(
                d_model=d_model, num_heads=4, num_layers=2,
                ff_expansion=2, conv_kernel=31, dropout=0.3)
        else:
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
                   and optionally 'boundary_logits' if boundary head is enabled
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

        out = {
            "logits_v": logits_v,
            "logits_g": logits_g,
            "lvf": lvf,
            "gcf": gcf,
        }

        # Boundary head output (only when enabled)
        if self.visual_module.use_boundary_head and self.visual_module.boundary_head is not None:
            boundary_logits = self.visual_module.boundary_head(lvf)  # (B, T', 2)
            out["boundary_logits"] = boundary_logits

        return out

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Greedy CTC decoding using contextual module output."""
        with torch.no_grad():
            out = self.forward(x)
            log_probs = F.log_softmax(out["logits_g"], dim=-1)
            # (B, T', C+1) → argmax → remove blanks and repeats
            pred = log_probs.argmax(dim=-1)            # (B, T')
        return pred
