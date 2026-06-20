# ImageNet Pretrained Backbone

**Experiment:** `model_pretrained` | **Category:** Model Enhancement

## Overview

Uses ImageNet-pretrained ResNet-18 weights instead of random initialization. Full fine-tuning of all layers. The pretrained features provide a strong spatial prior for sign language frames.

| Property | Value |
|---|---|
| Parameters | 26.0M |
| Trainable | 26.0M |
| Frozen | 0 |
| Enhancements | ImageNet pretrained ResNet-18 (full fine-tune) |
| Dev WER | **20.96%** |
| Test WER | **21.48%** |
| Δ from baseline | **-0.92%** |
| Est. total time | **34.0h** (1.4d) |

---

## 📊 WER Progression

| Epoch | Stage | WER (dev) |
|---|---|---|
| 5 | 1 | 82.50% |
| 10 | 1 | 72.35% |
| 15 | 1 | 63.63% |
| 20 | 1 | 56.40% |
| 25 | 1 | 50.60% |
| 30 | 1 | 46.88% |
| 35 | 2 | 41.09% |
| 40 | 2 | 37.76% |
| 45 | 2 | 34.74% |
| 50 | 2 | 31.91% |
| 55 | 2 | 29.89% |
| 60 | 2 | 28.22% |
| 65 | 2 | 26.72% |
| 70 | 2 | 25.61% |
| 75 | 2 | 24.45% |
| 80 | 2 | 23.61% |
| 85 | 2 | 22.45% |
| 90 | 2 | 23.62% |
| 95 | 3 | 21.44% |
| 100 | 3 | 20.96% |

---

## 📉 Training Loss Curves

| Epoch | St | CTC(g) | CTC(v) | Seg | Total |
|---|---|---|---|---|---|
| 5 | 1 | 2.469 | 2.780 | — | 3.860 |
| 10 | 1 | 1.796 | 2.078 | — | 2.829 |
| 15 | 1 | 1.453 | 1.719 | — | 2.324 |
| 20 | 1 | 1.284 | 1.547 | — | 2.066 |
| 25 | 1 | 1.207 | 1.465 | — | 1.938 |
| 30 | 1 | 1.165 | 1.421 | — | 1.874 |
| 35 | 2 | 1.087 | — | 3.890 | 3.028 |
| 40 | 2 | 0.925 | — | 3.284 | 2.595 |
| 45 | 2 | 0.812 | — | 2.841 | 2.256 |
| 50 | 2 | 0.731 | — | 2.493 | 2.020 |
| 55 | 2 | 0.684 | — | 2.261 | 1.841 |
| 60 | 2 | 0.642 | — | 2.094 | 1.711 |
| 65 | 2 | 0.615 | — | 1.983 | 1.646 |
| 70 | 2 | 0.594 | — | 1.914 | 1.581 |
| 75 | 2 | 0.578 | — | 1.862 | 1.555 |
| 80 | 2 | 0.565 | — | 1.823 | 1.516 |
| 85 | 2 | 0.564 | — | 1.797 | 1.491 |
| 90 | 2 | 0.567 | — | 1.767 | 1.467 |
| 95 | 3 | 0.499 | — | — | 0.510 |
| 100 | 3 | 0.462 | — | — | 0.489 |

---

## 📋 Per-Stage Summary

| Stage | Mode | Epochs | CTC(g) | CTC(v) | Seg | Total |
|---|---|---|---|---|---|---|
| 1 | Synchronous | 1–30 | 1.713 | 1.992 | — | 2.713 |
| 2 | GSBA | 31–90 | 0.721 | — | 2.429 | 1.960 |
| 3 | Decouple | 91–100 | 0.516 | — | — | 0.523 |

---

## ⏱ Timing Profile (per epoch)

| Phase | Time | % |
|---|---|---|
| Data transfer | 37s | 3.0% |
| Forward pass | 713s | 58.3% |
| GSBA | 71s | 5.8% |
| Loss compute | 22s | 1.8% |
| Backward+Opt | 381s | 31.1% |
| **Total** | **1224s** | **100%** |

> Forward pass dominates (58.3%) — the ResNet-18 spatial encoder processes each frame independently.

---

## 🔬 Analysis

- **Stage 1 (1–30):** Both CTC(g) and CTC(v) decrease rapidly as the shared classifier learns.  
  CTC(g) drops from 3.45 → 1.17.
- **Stage 2 (31–90):** GSBA replaces CTC(v). Initial seg loss is high (4.53)  
  but decreases as pseudo-labels improve. CTC(g) continues to decrease.
- **Stage 3 (91–100):** Decouple — visual branch detached, only CTC(g) trains.  
  Small WER spike at transition (+-1.16%) then stabilizes.

**WER vs Baseline:** -0.92% — **improvement** ✅
