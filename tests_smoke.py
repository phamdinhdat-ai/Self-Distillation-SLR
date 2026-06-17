"""Quick smoke test — verifies all modules import and run correctly."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from models.smkd import SMKD
from losses import SMKDLoss, gsba
from dataset import PhoenixDataset

def test_smkd():
    model = SMKD(num_classes=20, d_model=32, hidden_size=32, backbone='resnet18',
                 pretrained_backbone=False, freeze_backbone=False)
    model.set_stage(1)
    x = torch.randn(2, 8, 3, 64, 64)
    out = model(x)
    assert out["logits_v"].shape == (2, 4, 21), f"Bad shape: {out['logits_v'].shape}"
    print(f"  [PASS] SMKD forward")

def test_criterion():
    criterion = SMKDLoss(blank=0, alpha=0.5, feature_alignment_enabled=True,
                         entropy_reg_enabled=True, focal_ctc_enabled=True,
                         gsba_confidence_weight=True)
    B, T, C = 2, 4, 21
    lv = torch.randn(B, T, C); lg = torch.randn(B, T, C)
    lvf = torch.randn(B, T, 32); gcf = torch.randn(B, T, 32)
    tgt = torch.tensor([1,2,3,4], dtype=torch.long)
    il = torch.tensor([T,T]); tl = torch.tensor([2,2])
    seg = torch.randint(-1, 19, (B, T)); mar = torch.rand(B, T)

    for stage, sname in [(1, "sync"), (2, "GSBA"), (3, "decouple")]:
        kwargs = {}
        if stage == 2:
            kwargs = {"seg_labels": seg, "seg_margins": mar}
        loss, d = criterion(lv, lg, tgt, il, tl, stage=stage, lvf=lvf, gcf=gcf, **kwargs)
        keys = list(d.keys())
        assert "ctc_g" in keys
        assert "total" in keys
        print(f"  [PASS] SMKDLoss stage {stage} ({sname}): keys={keys}")

def test_gsba():
    T, dm = 20, 32
    g = torch.randn(T, dm)
    w = torch.nn.functional.normalize(torch.randn(21, dm), dim=-1)
    sp = torch.zeros(T, dtype=torch.long); sp[3]=2; sp[8]=5
    gl = [2,5]
    seg, margins = gsba(g, w, sp, gl, d=2, return_margins=True)
    assert seg[3] == 2
    assert seg[8] == 5
    print(f"  [PASS] GSBA with margins")

def test_dataset():
    ds = PhoenixDataset(root='.', split='train', synthetic=True, cache_dir='/tmp/test_cache_smkd')
    item = ds[0]
    assert "frames" in item and "labels" in item
    ds2 = PhoenixDataset(root='.', split='train', synthetic=True)
    item2 = ds2[0]
    print(f"  [PASS] Dataset (cache + no-cache)")

if __name__ == "__main__":
    print("SMKD Smoke Tests")
    print("-" * 40)
    test_smkd()
    test_criterion()
    test_gsba()
    test_dataset()
    print("-" * 40)
    print("ALL TESTS PASSED")
