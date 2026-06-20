# Baseline (Paper Reproduction)

**Experiment:** `baseline` | **Category:** Reference

## Overview

SMKD paper reproduction — ResNet-18 + C5-P2-C5 + BiLSTM. No enhancements. Three-stage training (30/90/100).

| Property | Value |
|---|---|
| Parameters | 26.0M |
| Trainable | 26.0M |
| Frozen | 0 |
| Enhancements | None — paper baseline |
| Dev WER | **21.88%** |
| Test WER | **22.56%** |
| Δ from baseline | **+0.00%** |
| Est. total time | **34.0h** (1.4d) |

---

## 📊 WER Progression

| Epoch | Stage | WER (dev) |
|---|---|---|
| 5 | 1 | 82.66% |
| 10 | 1 | 72.43% |
| 15 | 1 | 63.60% |
| 20 | 1 | 56.86% |
| 25 | 1 | 50.83% |
| 30 | 1 | 47.39% |
| 35 | 2 | 41.73% |
| 40 | 2 | 38.15% |
| 45 | 2 | 35.01% |
| 50 | 2 | 32.70% |
| 55 | 2 | 31.02% |
| 60 | 2 | 28.92% |
| 65 | 2 | 27.49% |
| 70 | 2 | 26.19% |
| 75 | 2 | 25.15% |
| 80 | 2 | 23.91% |
| 85 | 2 | 23.23% |
| 90 | 2 | 24.83% |
| 95 | 3 | 22.10% |
| 100 | 3 | 21.88% |

---

## 📉 Training Loss Curves

| Epoch | St | CTC(g) | CTC(v) | Seg | Total |
|---|---|---|---|---|---|
| 5 | 1 | 2.520 | 2.778 | — | 3.902 |
| 10 | 1 | 1.844 | 2.076 | — | 2.884 |
| 15 | 1 | 1.498 | 1.720 | — | 2.367 |
| 20 | 1 | 1.335 | 1.547 | — | 2.115 |
| 25 | 1 | 1.254 | 1.464 | — | 1.986 |
| 30 | 1 | 1.215 | 1.422 | — | 1.924 |
| 35 | 2 | 1.138 | — | 3.888 | 3.079 |
| 40 | 2 | 0.972 | — | 3.291 | 2.644 |
| 45 | 2 | 0.856 | — | 2.853 | 2.304 |
| 50 | 2 | 0.782 | — | 2.521 | 2.071 |
| 55 | 2 | 0.740 | — | 2.266 | 1.885 |
| 60 | 2 | 0.691 | — | 2.086 | 1.763 |
| 65 | 2 | 0.666 | — | 1.991 | 1.674 |
| 70 | 2 | 0.642 | — | 1.917 | 1.631 |
| 75 | 2 | 0.627 | — | 1.855 | 1.604 |
| 80 | 2 | 0.608 | — | 1.827 | 1.572 |
| 85 | 2 | 0.614 | — | 1.788 | 1.541 |
| 90 | 2 | 0.623 | — | 1.760 | 1.523 |
| 95 | 3 | 0.547 | — | — | 0.555 |
| 100 | 3 | 0.514 | — | — | 0.541 |

---

## 📋 Per-Stage Summary

| Stage | Mode | Epochs | CTC(g) | CTC(v) | Seg | Total |
|---|---|---|---|---|---|---|
| 1 | Synchronous | 1–30 | 1.763 | 1.995 | — | 2.761 |
| 2 | GSBA | 31–90 | 0.771 | — | 2.428 | 2.010 |
| 3 | Decouple | 91–100 | 0.566 | — | — | 0.575 |

---

## ⏱ Timing Profile (per epoch)

| Phase | Time | % |
|---|---|---|
| Data transfer | 35s | 2.9% |
| Forward pass | 706s | 57.7% |
| GSBA | 62s | 5.1% |
| Loss compute | 25s | 2.0% |
| Backward+Opt | 396s | 32.4% |
| **Total** | **1224s** | **100%** |

> Forward pass dominates (57.7%) — the ResNet-18 spatial encoder processes each frame independently.

---

## 🔬 Analysis

- **Stage 1 (1–30):** Both CTC(g) and CTC(v) decrease rapidly as the shared classifier learns.  
  CTC(g) drops from 3.51 → 1.22.
- **Stage 2 (31–90):** GSBA replaces CTC(v). Initial seg loss is high (4.54)  
  but decreases as pseudo-labels improve. CTC(g) continues to decrease.
- **Stage 3 (91–100):** Decouple — visual branch detached, only CTC(g) trains.  
  Small WER spike at transition (+-0.68%) then stabilizes.

**WER vs Baseline:** +0.00% — regression ⚠️
