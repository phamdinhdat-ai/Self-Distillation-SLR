# Combined Tier 2 (All Enhancements)

**Experiment:** `combined_tier2` | **Category:** Combined

## Overview

All enhancements together: VAC + Entropy + Focal CTC + Confidence-weighted GSBA + Curriculum α + Depthwise-separable convs + Multi-scale branches + AMP + EMA + Temporal aug + Smooth transitions. Best overall WER.

| Property | Value |
|---|---|
| Parameters | 28.1M |
| Trainable | 28.1M |
| Frozen | 0 |
| Enhancements | All Tier 1 + Tier 2 enhancements combined |
| Dev WER | **19.88%** |
| Test WER | **20.67%** |
| Δ from baseline | **-2.00%** |
| Est. total time | **9.4h** (0.4d) |

---

## 📊 WER Progression

| Epoch | Stage | WER (dev) |
|---|---|---|
| 5 | 1 | 82.35% |
| 10 | 1 | 71.77% |
| 15 | 1 | 63.15% |
| 20 | 1 | 56.04% |
| 25 | 1 | 49.88% |
| 30 | 1 | 46.27% |
| 35 | 2 | 40.70% |
| 40 | 2 | 36.73% |
| 45 | 2 | 33.80% |
| 50 | 2 | 31.26% |
| 55 | 2 | 29.08% |
| 60 | 2 | 27.03% |
| 65 | 2 | 25.04% |
| 70 | 2 | 24.28% |
| 75 | 2 | 23.18% |
| 80 | 2 | 22.19% |
| 85 | 2 | 21.38% |
| 90 | 2 | 22.72% |
| 95 | 3 | 20.43% |
| 100 | 3 | 19.88% |

---

## 📉 Training Loss Curves

| Epoch | St | CTC(g) | CTC(v) | Seg | Total |
|---|---|---|---|---|---|
| 5 | 1 | 2.469 | 2.779 | — | 3.853 |
| 10 | 1 | 1.793 | 2.076 | — | 2.849 |
| 15 | 1 | 1.453 | 1.719 | — | 2.335 |
| 20 | 1 | 1.287 | 1.543 | — | 2.071 |
| 25 | 1 | 1.207 | 1.461 | — | 1.932 |
| 30 | 1 | 1.168 | 1.420 | — | 1.880 |
| 35 | 2 | 1.094 | — | 3.881 | 3.039 |
| 40 | 2 | 0.924 | — | 3.289 | 2.597 |
| 45 | 2 | 0.812 | — | 2.842 | 2.267 |
| 50 | 2 | 0.736 | — | 2.503 | 2.023 |
| 55 | 2 | 0.688 | — | 2.249 | 1.844 |
| 60 | 2 | 0.640 | — | 2.094 | 1.715 |
| 65 | 2 | 0.606 | — | 1.983 | 1.649 |
| 70 | 2 | 0.591 | — | 1.910 | 1.586 |
| 75 | 2 | 0.576 | — | 1.873 | 1.545 |
| 80 | 2 | 0.561 | — | 1.816 | 1.526 |
| 85 | 2 | 0.566 | — | 1.793 | 1.498 |
| 90 | 2 | 0.571 | — | 1.762 | 1.490 |
| 95 | 3 | 0.501 | — | — | 0.505 |
| 100 | 3 | 0.464 | — | — | 0.488 |

---

## 📋 Per-Stage Summary

| Stage | Mode | Epochs | CTC(g) | CTC(v) | Seg | Total |
|---|---|---|---|---|---|---|
| 1 | Synchronous | 1–30 | 1.714 | 1.991 | — | 2.719 |
| 2 | GSBA | 31–90 | 0.721 | — | 2.429 | 1.965 |
| 3 | Decouple | 91–100 | 0.517 | — | — | 0.527 |

---

## ⏱ Timing Profile (per epoch)

| Phase | Time | % |
|---|---|---|
| Data transfer | 10s | 2.9% |
| Forward pass | 196s | 57.6% |
| GSBA | 20s | 5.9% |
| Loss compute | 6s | 1.8% |
| Backward+Opt | 108s | 31.8% |
| **Total** | **340s** | **100%** |

> Forward pass dominates (57.6%) — the ResNet-18 spatial encoder processes each frame independently.

---

## 🔬 Analysis

- **Stage 1 (1–30):** Both CTC(g) and CTC(v) decrease rapidly as the shared classifier learns.  
  CTC(g) drops from 3.46 → 1.17.
- **Stage 2 (31–90):** GSBA replaces CTC(v). Initial seg loss is high (4.54)  
  but decreases as pseudo-labels improve. CTC(g) continues to decrease.
- **Stage 3 (91–100):** Decouple — visual branch detached, only CTC(g) trains.  
  Small WER spike at transition (+-0.81%) then stabilizes.

**WER vs Baseline:** -2.00% — **improvement** ✅
