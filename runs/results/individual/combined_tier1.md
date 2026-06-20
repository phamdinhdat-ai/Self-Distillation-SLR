# Combined Tier 1

**Experiment:** `combined_tier1` | **Category:** Combined

## Overview

Combines the best practical enhancements: VAC alignment + Entropy regularization + AMP + EMA + Temporal augmentation + Smooth stage transitions. All work together with minimal individual overhead.

| Property | Value |
|---|---|
| Parameters | 26.0M |
| Trainable | 26.0M |
| Frozen | 0 |
| Enhancements | VAC + Entropy + AMP + EMA + Temporal Aug + Smooth Transitions |
| Dev WER | **20.59%** |
| Test WER | **21.54%** |
| Δ from baseline | **-1.28%** |
| Est. total time | **9.4h** (0.4d) |

---

## 📊 WER Progression

| Epoch | Stage | WER (dev) |
|---|---|---|
| 5 | 1 | 81.84% |
| 10 | 1 | 72.26% |
| 15 | 1 | 63.46% |
| 20 | 1 | 56.24% |
| 25 | 1 | 49.94% |
| 30 | 1 | 47.03% |
| 35 | 2 | 40.38% |
| 40 | 2 | 37.02% |
| 45 | 2 | 34.18% |
| 50 | 2 | 31.51% |
| 55 | 2 | 29.68% |
| 60 | 2 | 27.34% |
| 65 | 2 | 26.11% |
| 70 | 2 | 24.77% |
| 75 | 2 | 23.71% |
| 80 | 2 | 22.63% |
| 85 | 2 | 21.88% |
| 90 | 2 | 23.19% |
| 95 | 3 | 20.91% |
| 100 | 3 | 20.59% |

---

## 📉 Training Loss Curves

| Epoch | St | CTC(g) | CTC(v) | Seg | Total |
|---|---|---|---|---|---|
| 5 | 1 | 2.480 | 2.777 | — | 3.894 |
| 10 | 1 | 1.818 | 2.074 | — | 2.860 |
| 15 | 1 | 1.475 | 1.722 | — | 2.348 |
| 20 | 1 | 1.307 | 1.540 | — | 2.084 |
| 25 | 1 | 1.224 | 1.460 | — | 1.952 |
| 30 | 1 | 1.195 | 1.426 | — | 1.904 |
| 35 | 2 | 1.104 | — | 3.882 | 3.060 |
| 40 | 2 | 0.942 | — | 3.288 | 2.625 |
| 45 | 2 | 0.832 | — | 2.851 | 2.290 |
| 50 | 2 | 0.754 | — | 2.501 | 2.048 |
| 55 | 2 | 0.710 | — | 2.253 | 1.866 |
| 60 | 2 | 0.658 | — | 2.077 | 1.730 |
| 65 | 2 | 0.636 | — | 1.986 | 1.645 |
| 70 | 2 | 0.612 | — | 1.916 | 1.607 |
| 75 | 2 | 0.597 | — | 1.856 | 1.576 |
| 80 | 2 | 0.580 | — | 1.827 | 1.544 |
| 85 | 2 | 0.586 | — | 1.805 | 1.511 |
| 90 | 2 | 0.590 | — | 1.759 | 1.489 |
| 95 | 3 | 0.521 | — | — | 0.530 |
| 100 | 3 | 0.487 | — | — | 0.518 |

---

## 📋 Per-Stage Summary

| Stage | Mode | Epochs | CTC(g) | CTC(v) | Seg | Total |
|---|---|---|---|---|---|---|
| 1 | Synchronous | 1–30 | 1.734 | 1.991 | — | 2.738 |
| 2 | GSBA | 31–90 | 0.741 | — | 2.428 | 1.983 |
| 3 | Decouple | 91–100 | 0.538 | — | — | 0.549 |

---

## ⏱ Timing Profile (per epoch)

| Phase | Time | % |
|---|---|---|
| Data transfer | 11s | 3.2% |
| Forward pass | 193s | 56.8% |
| GSBA | 18s | 5.3% |
| Loss compute | 7s | 2.1% |
| Backward+Opt | 111s | 32.6% |
| **Total** | **340s** | **100%** |

> Forward pass dominates (56.8%) — the ResNet-18 spatial encoder processes each frame independently.

---

## 🔬 Analysis

- **Stage 1 (1–30):** Both CTC(g) and CTC(v) decrease rapidly as the shared classifier learns.  
  CTC(g) drops from 3.48 → 1.19.
- **Stage 2 (31–90):** GSBA replaces CTC(v). Initial seg loss is high (4.53)  
  but decreases as pseudo-labels improve. CTC(g) continues to decrease.
- **Stage 3 (91–100):** Decouple — visual branch detached, only CTC(g) trains.  
  Small WER spike at transition (+-0.75%) then stabilizes.

**WER vs Baseline:** -1.28% — **improvement** ✅
