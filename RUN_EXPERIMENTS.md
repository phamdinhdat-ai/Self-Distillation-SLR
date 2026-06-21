
---

## 1. Smoke Test (validate all new code works)

```bash
python tests_smoke_new.py
```

This tests all 6 enhancements in one shot — Conformer, Boundary Head, Proto Contrastive, Spike Penalty, and enhanced SMKDLoss.

---

## 2. Quick Synthetic Training (tiny model, 2 min)

```bash
python train.py --synthetic --epochs 6 --d_model 32 --hidden_size 32 --img_size 32 --max_frames 16 --stage1_end 2 --stage2_end 4 --eval_every 1
```

This runs a full 3-stage pipeline on random data with a mini model — great for verifying no crashes in the training loop.

---

## 3. With a Config File (test specific enhancements)

```bash
# Test the original paper baseline
python train.py --synthetic --config configs/experiments/baseline.yaml --epochs 6 --d_model 32 --hidden_size 32 --stage1_end 2 --stage2_end 4 --eval_every 1

# Test all enhancements combined
python train.py --synthetic --config configs/experiments/combined_tier3.yaml --epochs 6 --d_model 32 --hidden_size 32 --stage1_end 2 --stage2_end 4 --eval_every 1

# Test individual ones
python train.py --synthetic --config configs/experiments/model_conformer.yaml --epochs 6 --d_model 32 --hidden_size 32 --stage1_end 2 --stage2_end 4 --eval_every 1
python train.py --synthetic --config configs/experiments/loss_proto_contrastive.yaml --epochs 6 --d_model 32 --hidden_size 32 --stage1_end 2 --stage2_end 4 --eval_every 1
```

---

## 4. Full Training (paper baseline, 100 epochs)

```bash
python train.py --synthetic --epochs 100
```

Or with real data (requires PHOENIX14-T at `./data/phoenix`):

```bash
python train.py --real --data_root ./data/phoenix --epochs 100
```

---

> **Note**: CLI args (`--epochs`, `--d_model`, etc.) always override YAML config values — so you can mix and match. Your .venv is already activated, so all commands should work directly.