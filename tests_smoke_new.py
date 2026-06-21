"""Quick smoke test for all new enhancements."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from models.smkd import (
    SMKD, VisualModule, ContextualModule, NormalisedClassifier,
    TransformerContextualModule, ConformerContextualModule,
    ConformerConvModule, GlossBoundaryDetector,
)
from losses import (
    SMKDLoss, batch_gsba, CTCLoss, gsba, FocalCTCLoss,
    prototype_contrastive_loss, blank_variance_penalty,
)

print("[PASS] All imports OK")

# ---- Test Conformer ----
model_cf = SMKD(num_classes=20, d_model=64, hidden_size=64, backbone='resnet18',
                context_type='conformer')
model_cf.set_stage(1)
x = torch.randn(2, 16, 3, 64, 64)
out = model_cf(x)
assert out["logits_v"].shape == (2, 8, 21), f"Bad shape: {out['logits_v'].shape}"
assert out["logits_g"].shape == (2, 8, 21)
print(f"[PASS] Conformer forward OK")

# ---- Test Boundary Head ----
model_bd = SMKD(num_classes=20, d_model=64, hidden_size=64, use_boundary_head=True)
model_bd.set_stage(1)
out2 = model_bd(x)
assert "boundary_logits" in out2
assert out2["boundary_logits"].shape == (2, 8, 2)
print(f"[PASS] Boundary head forward OK")

# ---- Test Proto Contrastive ----
w = torch.randn(21, 64)
l_proto = prototype_contrastive_loss(w, blank=0, margin=0.3)
print(f"[PASS] Proto contrastive: {l_proto.item():.4f}")

# ---- Test Variance Spike Penalty ----
logits = torch.randn(2, 20, 21)
l_spike = blank_variance_penalty(logits, blank=0)
print(f"[PASS] Spike penalty: {l_spike.item():.4f}")

# ---- Test GlossBoundaryDetector ----
bd = GlossBoundaryDetector(64)
lvf = torch.randn(2, 10, 64)
b_out = bd(lvf)
assert b_out.shape == (2, 10, 2)
print(f"[PASS] Boundary detector: {b_out.shape}")

# ---- Test ConformerConvModule ----
ccm = ConformerConvModule(64, kernel_size=7)
conv_out = ccm(torch.randn(2, 10, 64))
assert conv_out.shape == (2, 10, 64)
print(f"[PASS] ConformerConvModule: {conv_out.shape}")

# ---- Test Enhanced SMKDLoss ----
criterion = SMKDLoss(
    blank=0, alpha=0.5,
    feature_alignment_enabled=True,
    entropy_reg_enabled=True,
    proto_contrastive_enabled=True,
    proto_contrastive_weight=0.01,
    proto_contrastive_margin=0.3,
    spike_penalty_enabled=True,
    spike_penalty_weight=0.01,
)
B, T, C = 2, 10, 21
logits_v = torch.randn(B, T, C)
logits_g = torch.randn(B, T, C)
lvf_t = torch.randn(B, T, 64)
gcf_t = torch.randn(B, T, 64)
targets = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.long)
il = torch.tensor([T, T], dtype=torch.long)
tl = torch.tensor([3, 3], dtype=torch.long)
seg = torch.randint(-1, 19, (B, T))
margins = torch.rand(B, T)

criterion._proto_weight_matrix = torch.randn(21, 64)
loss, d = criterion(logits_v, logits_g, targets, il, tl, stage=1, lvf=lvf_t, gcf=gcf_t)
print(f"[PASS] Enhanced SMKDLoss stage 1: total={d['total']:.4f}, proto={d.get('proto',0):.4f}, spike={d.get('spike',0):.4f}")

loss2, d2 = criterion(logits_v, logits_g, targets, il, tl, stage=2, seg_labels=seg, seg_margins=margins, lvf=lvf_t, gcf=gcf_t)
print(f"[PASS] Enhanced SMKDLoss stage 2: total={d2['total']:.4f}")

# ---- Test boundary label derivation ----
from trainer import _derive_boundary_labels
seg_test = torch.tensor([
    [1, 1, 2, 2, 2, 3, -1, -1],
    [4, 4, 4, 5, 5, -1, -1, -1],
])
bd_labels = _derive_boundary_labels(seg_test)
print(f"[PASS] Boundary labels: {bd_labels.tolist()}")

print()
print("All smoke tests passed!")
