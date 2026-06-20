# Transformer Contextual Module

**Experiment:** `model_transformer` | **Category:** Model Enhancement

## Overview

Replaces the 2-layer BiLSTM with a lightweight 2-layer Transformer encoder (4 heads, FFN×2, Pre-LN). Better GPU parallelism than BiLSTM — processes all timesteps simultaneously. Similar param count (~10.3M vs ~10.5M). Positional encoding provides temporal awareness.

| Property | Value |
|---|---|
| Parameters | 25.8M |
| Trainable | 25.8M |
| Frozen | 0 |
| Enhancements | Transformer encoder (4 heads, 2 layers) + ImageNet pretrained backbone |
| Dev WER | **21.31%** |
| Test WER | **21.86%** |
| Δ from baseline | **-0.57%** |
| Est. total time | **27.2h** (1.1d) |

---

## 📊 WER Progression

| Epoch | Stage | WER (dev) |
|---|---|---|
| 5 | 1 | 82.53% |
| 10 | 1 | 72.29% |
| 15 | 1 | 63.78% |
| 20 | 1 | 56.82% |
| 25 | 1 | 50.79% |
| 30 | 1 | 47.24% |
| 35 | 2 | 41.44% |
| 40 | 2 | 37.78% |
| 45 | 2 | 34.70% |
| 50 | 2 | 32.12% |
| 55 | 2 | 30.19% |
| 60 | 2 | 28.40% |
| 65 | 2 | 26.58% |
| 70 | 2 | 25.31% |
| 75 | 2 | 24.44% |
| 80 | 2 | 23.75% |
| 85 | 2 | 22.69% |
| 90 | 2 | 24.30% |
| 95 | 3 | 21.80% |
| 100 | 3 | 21.31% |

---

## 📉 Training Loss Curves

| Epoch | St | CTC(g) | CTC(v) | Seg | Total |
|---|---|---|---|---|---|
| 5 | 1 | 2.499 | 2.775 | — | 3.891 |
| 10 | 1 | 1.825 | 2.081 | — | 2.873 |
| 15 | 1 | 1.484 | 1.720 | — | 2.355 |
| 20 | 1 | 1.319 | 1.544 | — | 2.086 |
| 25 | 1 | 1.238 | 1.464 | — | 1.960 |
| 30 | 1 | 1.199 | 1.429 | — | 1.903 |
| 35 | 2 | 1.120 | — | 3.887 | 3.066 |
| 40 | 2 | 0.953 | — | 3.284 | 2.631 |
| 45 | 2 | 0.839 | — | 2.843 | 2.287 |
| 50 | 2 | 0.761 | — | 2.502 | 2.047 |
| 55 | 2 | 0.716 | — | 2.281 | 1.870 |
| 60 | 2 | 0.672 | — | 2.090 | 1.743 |
| 65 | 2 | 0.640 | — | 1.993 | 1.667 |
| 70 | 2 | 0.617 | — | 1.917 | 1.614 |
| 75 | 2 | 0.605 | — | 1.869 | 1.576 |
| 80 | 2 | 0.594 | — | 1.835 | 1.546 |
| 85 | 2 | 0.595 | — | 1.785 | 1.516 |
| 90 | 2 | 0.604 | — | 1.772 | 1.501 |
| 95 | 3 | 0.531 | — | — | 0.528 |
| 100 | 3 | 0.494 | — | — | 0.517 |

---

## 📋 Per-Stage Summary

| Stage | Mode | Epochs | CTC(g) | CTC(v) | Seg | Total |
|---|---|---|---|---|---|---|
| 1 | Synchronous | 1–30 | 1.745 | 1.993 | — | 2.742 |
| 2 | GSBA | 31–90 | 0.751 | — | 2.429 | 1.989 |
| 3 | Decouple | 91–100 | 0.549 | — | — | 0.552 |

---

## ⏱ Timing Profile (per epoch)

| Phase | Time | % |
|---|---|---|
| Data transfer | 24s | 2.5% |
| Forward pass | 550s | 56.2% |
| GSBA | 64s | 6.5% |
| Loss compute | 21s | 2.1% |
| Backward+Opt | 320s | 32.7% |
| **Total** | **979s** | **100%** |

> Forward pass dominates (56.2%) — the ResNet-18 spatial encoder processes each frame independently.

---

## 🔬 Analysis

- **Stage 1 (1–30):** Both CTC(g) and CTC(v) decrease rapidly as the shared classifier learns.  
  CTC(g) drops from 3.48 → 1.20.
- **Stage 2 (31–90):** GSBA replaces CTC(v). Initial seg loss is high (4.53)  
  but decreases as pseudo-labels improve. CTC(g) continues to decrease.
- **Stage 3 (91–100):** Decouple — visual branch detached, only CTC(g) trains.  
  Small WER spike at transition (+-1.06%) then stabilizes.

**WER vs Baseline:** -0.57% — **improvement** ✅
