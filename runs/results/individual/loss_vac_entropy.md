# VAC Alignment + Entropy Regularization

**Experiment:** `loss_vac_entropy` | **Category:** Loss Enhancement

## Overview

Adds two complementary loss terms: (1) VAC-style cosine alignment between LVF and GCF at GSBA anchor frames, and (2) entropy maximization on the blank posterior to suppress CTC blank spikes.

| Property | Value |
|---|---|
| Parameters | 26.0M |
| Trainable | 26.0M |
| Frozen | 0 |
| Enhancements | VAC alignment (cosine, w=0.1) + Entropy regularization (w=0.01) |
| Dev WER | **21.45%** |
| Test WER | **22.36%** |
| Δ from baseline | **-0.43%** |
| Est. total time | **34.0h** (1.4d) |

---

## 📊 WER Progression

| Epoch | Stage | WER (dev) |
|---|---|---|
| 5 | 1 | 82.42% |
| 10 | 1 | 72.33% |
| 15 | 1 | 63.85% |
| 20 | 1 | 56.41% |
| 25 | 1 | 50.63% |
| 30 | 1 | 47.20% |
| 35 | 2 | 41.26% |
| 40 | 2 | 37.75% |
| 45 | 2 | 34.96% |
| 50 | 2 | 32.44% |
| 55 | 2 | 30.16% |
| 60 | 2 | 28.25% |
| 65 | 2 | 26.68% |
| 70 | 2 | 25.42% |
| 75 | 2 | 24.71% |
| 80 | 2 | 23.45% |
| 85 | 2 | 22.82% |
| 90 | 2 | 24.15% |
| 95 | 3 | 21.48% |
| 100 | 3 | 21.45% |

---

## 📉 Training Loss Curves

| Epoch | St | CTC(g) | CTC(v) | Seg | Total |
|---|---|---|---|---|---|
| 5 | 1 | 2.517 | 2.774 | — | 3.922 |
| 10 | 1 | 1.846 | 2.077 | — | 2.896 |
| 15 | 1 | 1.506 | 1.724 | — | 2.371 |
| 20 | 1 | 1.333 | 1.542 | — | 2.121 |
| 25 | 1 | 1.257 | 1.461 | — | 1.990 |
| 30 | 1 | 1.219 | 1.420 | — | 1.935 |
| 35 | 2 | 1.139 | — | 3.898 | 3.088 |
| 40 | 2 | 0.974 | — | 3.291 | 2.651 |
| 45 | 2 | 0.864 | — | 2.848 | 2.330 |
| 50 | 2 | 0.787 | — | 2.510 | 2.080 |
| 55 | 2 | 0.737 | — | 2.263 | 1.892 |
| 60 | 2 | 0.691 | — | 2.093 | 1.764 |
| 65 | 2 | 0.663 | — | 1.974 | 1.687 |
| 70 | 2 | 0.640 | — | 1.925 | 1.635 |
| 75 | 2 | 0.631 | — | 1.874 | 1.600 |
| 80 | 2 | 0.611 | — | 1.830 | 1.574 |
| 85 | 2 | 0.618 | — | 1.790 | 1.544 |
| 90 | 2 | 0.623 | — | 1.756 | 1.527 |
| 95 | 3 | 0.548 | — | — | 0.556 |
| 100 | 3 | 0.518 | — | — | 0.545 |

---

## 📋 Per-Stage Summary

| Stage | Mode | Epochs | CTC(g) | CTC(v) | Seg | Total |
|---|---|---|---|---|---|---|
| 1 | Synchronous | 1–30 | 1.765 | 1.991 | — | 2.772 |
| 2 | GSBA | 31–90 | 0.772 | — | 2.430 | 2.015 |
| 3 | Decouple | 91–100 | 0.567 | — | — | 0.574 |

---

## ⏱ Timing Profile (per epoch)

| Phase | Time | % |
|---|---|---|
| Data transfer | 31s | 2.5% |
| Forward pass | 709s | 57.9% |
| GSBA | 76s | 6.2% |
| Loss compute | 23s | 1.9% |
| Backward+Opt | 385s | 31.5% |
| **Total** | **1224s** | **100%** |

> Forward pass dominates (57.9%) — the ResNet-18 spatial encoder processes each frame independently.

---

## 🔬 Analysis

- **Stage 1 (1–30):** Both CTC(g) and CTC(v) decrease rapidly as the shared classifier learns.  
  CTC(g) drops from 3.50 → 1.22.
- **Stage 2 (31–90):** GSBA replaces CTC(v). Initial seg loss is high (4.55)  
  but decreases as pseudo-labels improve. CTC(g) continues to decrease.
- **Stage 3 (91–100):** Decouple — visual branch detached, only CTC(g) trains.  
  Small WER spike at transition (+-0.63%) then stabilizes.

**WER vs Baseline:** -0.43% — **improvement** ✅
