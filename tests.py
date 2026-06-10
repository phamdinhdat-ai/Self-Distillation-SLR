"""
Unit tests – run from the smkd/ directory:
    python tests.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn.functional as F

from models.smkd   import SMKD, VisualModule, ContextualModule, NormalisedClassifier
from models.losses import CTCLoss, LabelSmoothingCE, SMKDLoss, gsba, batch_gsba
from data.dataset  import Vocabulary, PhoenixDataset, collate_fn
from metrics       import compute_wer, edit_distance


# ---- helpers ---------------------------------------------------------------

def make_batch(B=2, T=32, C=3, H=64, W=64):
    return torch.randn(B, T, C, H, W)

BLANK      = 0
NUM_CLASSES = 20   # small vocabulary for tests


# ============================================================================
# 1. Model components
# ============================================================================

def test_visual_module():
    vm  = VisualModule(d_model=64)
    x   = make_batch(B=2, T=16, C=3, H=64, W=64)
    lvf = vm(x)
    # T' = T // 2 = 8
    assert lvf.shape == (2, 8, 64), f"Bad shape: {lvf.shape}"
    print("[PASS] VisualModule")


def test_contextual_module():
    cm  = ContextualModule(d_model=64, hidden_size=64)
    lvf = torch.randn(2, 8, 64)
    gcf = cm(lvf)
    assert gcf.shape == lvf.shape, f"Bad shape: {gcf.shape}"
    print("[PASS] ContextualModule")


def test_normalised_classifier():
    clf    = NormalisedClassifier(d_model=64, num_classes=NUM_CLASSES)
    feat   = torch.randn(2, 8, 64)
    logits = clf(feat)
    assert logits.shape == (2, 8, NUM_CLASSES + 1), f"Bad shape: {logits.shape}"
    print("[PASS] NormalisedClassifier")


def test_smkd_forward_stage1():
    model = SMKD(num_classes=NUM_CLASSES, d_model=64, hidden_size=64)
    model.set_stage(1)
    x = make_batch(B=2, T=16)
    out = model(x)
    assert "logits_v" in out and "logits_g" in out
    assert out["logits_v"].shape[-1] == NUM_CLASSES + 1
    print("[PASS] SMKD forward (stage 1)")


def test_smkd_forward_stage3():
    model = SMKD(num_classes=NUM_CLASSES, d_model=64, hidden_size=64)
    model.set_stage(3)
    x   = make_batch(B=2, T=16)
    out = model(x)
    # Stage 3 uses independent classifiers – shapes still the same
    assert out["logits_v"].shape == out["logits_g"].shape
    print("[PASS] SMKD forward (stage 3)")


# ============================================================================
# 2. Loss functions
# ============================================================================

def test_ctc_loss():
    loss_fn = CTCLoss(blank=BLANK)
    B, T, C = 2, 10, NUM_CLASSES + 1
    logits       = torch.randn(B, T, C)
    targets      = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.long)
    in_lengths   = torch.tensor([T, T], dtype=torch.long)
    tgt_lengths  = torch.tensor([3, 3], dtype=torch.long)
    loss = loss_fn(logits, targets, in_lengths, tgt_lengths)
    assert loss.item() > 0
    print("[PASS] CTCLoss")


def test_label_smoothing_ce():
    loss_fn = LabelSmoothingCE(smoothing=0.1, ignore_index=-1)
    logits  = torch.randn(10, NUM_CLASSES + 1)
    targets = torch.randint(0, NUM_CLASSES, (10,))
    targets[3] = -1   # ignored
    loss = loss_fn(logits, targets)
    assert loss.item() > 0
    print("[PASS] LabelSmoothingCE")


def test_smkd_loss_stage1():
    criterion = SMKDLoss(blank=0, alpha=0.5)
    B, T, C   = 2, 10, NUM_CLASSES + 1
    logits_v  = torch.randn(B, T, C)
    logits_g  = torch.randn(B, T, C)
    targets   = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.long)
    il        = torch.tensor([T, T], dtype=torch.long)
    tl        = torch.tensor([3, 3], dtype=torch.long)
    loss, d = criterion(logits_v, logits_g, targets, il, tl, stage=1)
    assert "ctc_g" in d and "ctc_v" in d
    print("[PASS] SMKDLoss stage 1")


def test_smkd_loss_stage2():
    criterion  = SMKDLoss(blank=0, alpha=0.5)
    B, T, C    = 2, 10, NUM_CLASSES + 1
    logits_v   = torch.randn(B, T, C)
    logits_g   = torch.randn(B, T, C)
    seg_labels = torch.randint(-1, NUM_CLASSES, (B, T))
    targets    = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.long)
    il         = torch.tensor([T, T], dtype=torch.long)
    tl         = torch.tensor([3, 3], dtype=torch.long)
    loss, d = criterion(logits_v, logits_g, targets, il, tl, stage=2, seg_labels=seg_labels)
    assert "seg" in d
    print("[PASS] SMKDLoss stage 2")


# ============================================================================
# 3. GSBA
# ============================================================================

def test_gsba():
    T, d_model = 20, 16
    gcf      = torch.randn(T, d_model)
    w        = F.normalize(torch.randn(NUM_CLASSES + 1, d_model), dim=-1)
    # Spike at frames 3, 8, 14  (non-blank)
    spike    = torch.zeros(T, dtype=torch.long)
    spike[3] = 2; spike[8] = 5; spike[14] = 9
    gloss_seq = [2, 5, 9]

    seg = gsba(gcf, w, spike, gloss_seq, d=2)
    assert seg.shape == (T,)
    # Anchor frames must be labelled
    assert seg[3].item() == 2
    assert seg[8].item() == 5
    assert seg[14].item() == 9
    print("[PASS] GSBA single sample")


def test_batch_gsba():
    B, T, d_model = 3, 20, 16
    gcf      = torch.randn(B, T, d_model)
    w        = F.normalize(torch.randn(NUM_CLASSES + 1, d_model), dim=-1)
    spike    = torch.zeros(B, T, dtype=torch.long)
    spike[0, 5]  = 3
    spike[1, 10] = 7
    spike[2, 15] = 12
    gloss_seqs = [[3], [7], [12]]
    seg = batch_gsba(gcf, w, spike, gloss_seqs, d=1)
    assert seg.shape == (B, T)
    print("[PASS] batch_gsba")


# ============================================================================
# 4. Dataset / collate
# ============================================================================

def test_synthetic_dataset():
    ds = PhoenixDataset(root=".", split="train", synthetic=True)
    item = ds[0]
    assert "frames" in item and "labels" in item
    assert item["frames"].ndim == 4   # (T, C, H, W)
    print("[PASS] PhoenixDataset (synthetic)")


def test_collate_fn():
    ds    = PhoenixDataset(root=".", split="train", synthetic=True)
    batch = [ds[i] for i in range(3)]
    coll  = collate_fn(batch)
    assert coll["frames"].ndim == 5    # (B, T_max, C, H, W)
    assert coll["labels"].ndim == 1
    print("[PASS] collate_fn")


# ============================================================================
# 5. Metrics
# ============================================================================

def test_edit_distance():
    assert edit_distance([1,2,3], [1,2,3]) == 0
    assert edit_distance([1,2,3], [1,3])   == 1   # 1 deletion
    assert edit_distance([],      [1,2])   == 2
    print("[PASS] edit_distance")


def test_compute_wer():
    hyps = [[1,2,3], [4,5]]
    refs = [[1,2,3], [4,6]]
    wer  = compute_wer(hyps, refs)
    assert abs(wer - 20.0) < 1e-3, f"Got {wer}"   # 1 error / 5 ref tokens
    print("[PASS] compute_wer")


# ============================================================================
# 6. End-to-end forward + loss pass
# ============================================================================

def test_end_to_end():
    model     = SMKD(num_classes=NUM_CLASSES, d_model=64, hidden_size=64)
    criterion = SMKDLoss(blank=0, alpha=0.5)
    model.set_stage(1)

    B, T = 2, 16
    x           = make_batch(B=B, T=T, H=64, W=64)
    out         = model(x)
    T_prime     = out["logits_v"].size(1)
    in_lengths  = torch.full((B,), T_prime, dtype=torch.long)
    targets     = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.long)
    tgt_lengths = torch.tensor([3, 3], dtype=torch.long)

    loss, d = criterion(
        out["logits_v"], out["logits_g"],
        targets, in_lengths, tgt_lengths,
        stage=1,
    )
    loss.backward()
    print(f"[PASS] end-to-end (loss={loss.item():.4f})")


# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print(" SMKD Unit Tests")
    print("=" * 60)

    test_visual_module()
    test_contextual_module()
    test_normalised_classifier()
    test_smkd_forward_stage1()
    test_smkd_forward_stage3()

    test_ctc_loss()
    test_label_smoothing_ce()
    test_smkd_loss_stage1()
    test_smkd_loss_stage2()

    test_gsba()
    test_batch_gsba()

    test_synthetic_dataset()
    test_collate_fn()

    test_edit_distance()
    test_compute_wer()

    test_end_to_end()

    print("=" * 60)
    print(" All tests passed!")
    print("=" * 60)
