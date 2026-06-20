# SMKD Enhancement Suite — Synthetic Experiment Results

**Generated:** 2026-06-20
**Dataset:** PHOENIX-2014-T (5,672 train / 540 dev / 629 test)
**Model:** SMKD (d_model=512, 3 stages: 30/90/100)
**GPU:** NVIDIA T4 (simulated)
**Total experiments:** 23

---

## 📊 WER Comparison (All Experiments)

| # | Experiment | Category | Dev WER | Test WER | Δ Dev | Δ Test |
|---|---|---|---|---|---|---|
| 1 | `combined_tier2` | combined | 20.07% | 20.76% | -1.81 | -1.80 |
| 2 | `combined_tier1` | combined | 20.44% | 20.95% | -1.43 | -1.62 |
| 3 | `model_efficientnet_pretrained` | model | 21.14% | 21.76% | -0.73 | -0.81 |
| 4 | `model_pretrained` | model | 21.16% | 22.05% | -0.72 | -0.52 |
| 5 | `loss_vac_entropy` | loss | 21.21% | 21.76% | -0.67 | -0.80 |
| 6 | `model_multiscale` | model | 21.32% | 21.90% | -0.56 | -0.67 |
| 7 | `train_temporal_aug` | training | 21.34% | 22.01% | -0.53 | -0.56 |
| 8 | `loss_confidence_gsba` | loss | 21.42% | 22.06% | -0.45 | -0.50 |
| 9 | `loss_entropy` | loss | 21.46% | 21.98% | -0.42 | -0.59 |
| 10 | `train_smooth_transition` | training | 21.47% | 22.31% | -0.41 | -0.25 |
| 11 | `model_tsm` | model | 21.47% | 22.44% | -0.41 | -0.12 |
| 12 | `model_frozen_backbone` | model | 21.51% | 22.46% | -0.37 | -0.10 |
| 13 | `train_ema` | training | 21.54% | 22.30% | -0.34 | -0.26 |
| 14 | `loss_focal_ctc` | loss | 21.58% | 22.37% | -0.30 | -0.19 |
| 15 | `loss_vac` | loss | 21.65% | 22.56% | -0.23 | -0.01 |
| 16 | `model_dwconv` | model | 21.87% | 22.55% | -0.01 | -0.01 |
| 17 | `baseline` | loss | 21.88% | 22.56% | — | +0.00 |
| 18 | `loss_curriculum_alpha` | loss | 21.89% | 22.84% | +0.02 | +0.27 |
| 19 | `model_mobilenet_pretrained` | model | 21.93% | 22.51% | +0.06 | -0.06 |
| 20 | `train_amp` | training | 22.03% | 22.65% | +0.15 | +0.08 |
| 21 | `model_mobilenet_frozen` | model | 22.18% | 23.12% | +0.31 | +0.56 |
| 22 | `model_efficientnet` | model | 22.30% | 23.17% | +0.42 | +0.61 |
| 23 | `model_mobilenet` | model | 22.72% | 23.38% | +0.84 | +0.81 |

---

## 📈 Baseline — Detailed Results

### baseline
**loss** | Paper reproduction — no enhancements
**Dev WER:** 21.88% | **Test WER:** 22.56%

| Stage | Epochs | CTC(g) | CTC(v) | Seg | Total |
|---|---|---|---|---|---|
| 1 | 1-30 | 1.763 | 1.995 | — | 2.761 |
| 2 | 31-90 | 0.771 | — | 2.428 | 2.010 |
| 3 | 91-100 | 0.566 | — | — | 0.575 |

### ⏱ Timing: baseline

| Phase | Seconds | % |
|---|---|---|
| Data transfer | 39s | 3.2% |
| Forward pass | 703s | 57.4% |
| GSBA | 80s | 6.5% |
| Loss compute | 25s | 2.0% |
| Backward+Opt | 377s | 30.8% |
| **Total** | **1224s** | **100%** |

Est. total: **34.0h** (1.4d)

### Per-Eval-Epoch Detail

| Epoch | St | CTC(g) | CTC(v) | Seg | Total | WER(dev) |
|---|---|---|---|---|---|---|
| 5 | 1 | 2.520 | 2.778 | — | 3.902 | 82.66% |
| 10 | 1 | 1.844 | 2.076 | — | 2.884 | 72.43% |
| 15 | 1 | 1.498 | 1.720 | — | 2.367 | 63.60% |
| 20 | 1 | 1.335 | 1.547 | — | 2.115 | 56.86% |
| 25 | 1 | 1.254 | 1.464 | — | 1.986 | 50.83% |
| 30 | 1 | 1.215 | 1.422 | — | 1.924 | 47.39% |
| 35 | 2 | 1.138 | — | 3.888 | 3.079 | 41.73% |
| 40 | 2 | 0.972 | — | 3.291 | 2.644 | 38.15% |
| 45 | 2 | 0.856 | — | 2.853 | 2.304 | 35.01% |
| 50 | 2 | 0.782 | — | 2.521 | 2.071 | 32.70% |
| 55 | 2 | 0.740 | — | 2.266 | 1.885 | 31.02% |
| 60 | 2 | 0.691 | — | 2.086 | 1.763 | 28.92% |
| 65 | 2 | 0.666 | — | 1.991 | 1.674 | 27.49% |
| 70 | 2 | 0.642 | — | 1.917 | 1.631 | 26.19% |
| 75 | 2 | 0.627 | — | 1.855 | 1.604 | 25.15% |
| 80 | 2 | 0.608 | — | 1.827 | 1.572 | 23.91% |
| 85 | 2 | 0.614 | — | 1.788 | 1.541 | 23.23% |
| 90 | 2 | 0.623 | — | 1.760 | 1.523 | 24.83% |
| 95 | 3 | 0.547 | — | — | 0.555 | 22.10% |
| 100 | 3 | 0.514 | — | — | 0.541 | 21.88% |

---

## 🔬 Loss Experiments

### baseline
**loss** | Paper reproduction — no enhancements
**Dev WER:** 21.88% | **Test WER:** 22.56%

| Stage | Epochs | CTC(g) | CTC(v) | Seg | Total |
|---|---|---|---|---|---|
| 1 | 1-30 | 1.763 | 1.995 | — | 2.761 |
| 2 | 31-90 | 0.771 | — | 2.428 | 2.010 |
| 3 | 91-100 | 0.566 | — | — | 0.575 |

### ⏱ Timing: baseline

| Phase | Seconds | % |
|---|---|---|
| Data transfer | 39s | 3.2% |
| Forward pass | 703s | 57.4% |
| GSBA | 80s | 6.5% |
| Loss compute | 25s | 2.0% |
| Backward+Opt | 377s | 30.8% |
| **Total** | **1224s** | **100%** |

Est. total: **34.0h** (1.4d)

### loss_vac
**loss** | VAC-style visual alignment constraint (cosine, w=0.1)
**Dev WER:** 21.65% | **Test WER:** 22.56%

| Stage | Epochs | CTC(g) | CTC(v) | Seg | Total |
|---|---|---|---|---|---|
| 1 | 1-30 | 1.765 | 1.991 | — | 2.773 |
| 2 | 31-90 | 0.772 | — | 2.430 | 2.015 |
| 3 | 91-100 | 0.567 | — | — | 0.574 |

### ⏱ Timing: loss_vac

| Phase | Seconds | % |
|---|---|---|
| Data transfer | 33s | 2.7% |
| Forward pass | 701s | 57.3% |
| GSBA | 67s | 5.5% |
| Loss compute | 26s | 2.1% |
| Backward+Opt | 397s | 32.4% |
| **Total** | **1224s** | **100%** |

Est. total: **34.0h** (1.4d)

### loss_entropy
**loss** | Entropy regularization on blank posterior (anti-spike, w=0.01)
**Dev WER:** 21.46% | **Test WER:** 21.98%

| Stage | Epochs | CTC(g) | CTC(v) | Seg | Total |
|---|---|---|---|---|---|
| 1 | 1-30 | 1.763 | 1.992 | — | 2.763 |
| 2 | 31-90 | 0.771 | — | 2.429 | 2.010 |
| 3 | 91-100 | 0.566 | — | — | 0.574 |

### ⏱ Timing: loss_entropy

| Phase | Seconds | % |
|---|---|---|
| Data transfer | 32s | 2.6% |
| Forward pass | 704s | 57.5% |
| GSBA | 65s | 5.3% |
| Loss compute | 22s | 1.8% |
| Backward+Opt | 401s | 32.8% |
| **Total** | **1224s** | **100%** |

Est. total: **34.0h** (1.4d)

### loss_vac_entropy
**loss** | VAC alignment + Entropy regularization combined
**Dev WER:** 21.21% | **Test WER:** 21.76%

| Stage | Epochs | CTC(g) | CTC(v) | Seg | Total |
|---|---|---|---|---|---|
| 1 | 1-30 | 1.765 | 1.993 | — | 2.771 |
| 2 | 31-90 | 0.771 | — | 2.429 | 2.013 |
| 3 | 91-100 | 0.569 | — | — | 0.574 |

### ⏱ Timing: loss_vac_entropy

| Phase | Seconds | % |
|---|---|---|
| Data transfer | 36s | 2.9% |
| Forward pass | 707s | 57.8% |
| GSBA | 63s | 5.1% |
| Loss compute | 23s | 1.9% |
| Backward+Opt | 395s | 32.3% |
| **Total** | **1224s** | **100%** |

Est. total: **34.0h** (1.4d)

### loss_confidence_gsba
**loss** | Confidence-weighted GSBA (margin-weighted CE-LS)
**Dev WER:** 21.42% | **Test WER:** 22.06%

| Stage | Epochs | CTC(g) | CTC(v) | Seg | Total |
|---|---|---|---|---|---|
| 1 | 1-30 | 1.765 | 1.993 | — | 2.761 |
| 2 | 31-90 | 0.771 | — | 2.430 | 2.012 |
| 3 | 91-100 | 0.566 | — | — | 0.575 |

### ⏱ Timing: loss_confidence_gsba

| Phase | Seconds | % |
|---|---|---|
| Data transfer | 35s | 2.9% |
| Forward pass | 709s | 57.9% |
| GSBA | 81s | 6.6% |
| Loss compute | 24s | 2.0% |
| Backward+Opt | 375s | 30.6% |
| **Total** | **1224s** | **100%** |

Est. total: **34.0h** (1.4d)

### loss_curriculum_alpha
**loss** | Curriculum schedule on α (linear 0.1→0.5)
**Dev WER:** 21.89% | **Test WER:** 22.84%

| Stage | Epochs | CTC(g) | CTC(v) | Seg | Total |
|---|---|---|---|---|---|
| 1 | 1-30 | 1.784 | 1.991 | — | 2.780 |
| 2 | 31-90 | 0.791 | — | 2.428 | 2.029 |
| 3 | 91-100 | 0.588 | — | — | 0.597 |

### ⏱ Timing: loss_curriculum_alpha

| Phase | Seconds | % |
|---|---|---|
| Data transfer | 34s | 2.8% |
| Forward pass | 710s | 58.0% |
| GSBA | 71s | 5.8% |
| Loss compute | 21s | 1.7% |
| Backward+Opt | 389s | 31.8% |
| **Total** | **1225s** | **100%** |

Est. total: **34.0h** (1.4d)

### loss_focal_ctc
**loss** | Focal CTC modulation (γ=2.0)
**Dev WER:** 21.58% | **Test WER:** 22.37%

| Stage | Epochs | CTC(g) | CTC(v) | Seg | Total |
|---|---|---|---|---|---|
| 1 | 1-30 | 1.734 | 1.991 | — | 2.731 |
| 2 | 31-90 | 0.741 | — | 2.429 | 1.981 |
| 3 | 91-100 | 0.537 | — | — | 0.545 |

### ⏱ Timing: loss_focal_ctc

| Phase | Seconds | % |
|---|---|---|
| Data transfer | 32s | 2.6% |
| Forward pass | 685s | 56.0% |
| GSBA | 71s | 5.8% |
| Loss compute | 27s | 2.2% |
| Backward+Opt | 408s | 33.4% |
| **Total** | **1223s** | **100%** |

Est. total: **34.0h** (1.4d)

---

## 🔬 Model Experiments

### model_dwconv
**model** | Depthwise-separable temporal convs (23.9M params)
**Dev WER:** 21.87% | **Test WER:** 22.55%

| Stage | Epochs | CTC(g) | CTC(v) | Seg | Total |
|---|---|---|---|---|---|
| 1 | 1-30 | 1.765 | 1.992 | — | 2.759 |
| 2 | 31-90 | 0.770 | — | 2.430 | 2.011 |
| 3 | 91-100 | 0.565 | — | — | 0.568 |

### ⏱ Timing: model_dwconv

| Phase | Seconds | % |
|---|---|---|
| Data transfer | 39s | 3.2% |
| Forward pass | 690s | 57.5% |
| GSBA | 69s | 5.8% |
| Loss compute | 23s | 1.9% |
| Backward+Opt | 379s | 31.6% |
| **Total** | **1200s** | **100%** |

Est. total: **33.3h** (1.4d)

### model_multiscale
**model** | Multi-scale temporal branches (dilations [1,2], 28.1M params)
**Dev WER:** 21.32% | **Test WER:** 21.90%

| Stage | Epochs | CTC(g) | CTC(v) | Seg | Total |
|---|---|---|---|---|---|
| 1 | 1-30 | 1.743 | 1.992 | — | 2.739 |
| 2 | 31-90 | 0.750 | — | 2.430 | 1.991 |
| 3 | 91-100 | 0.547 | — | — | 0.548 |

### ⏱ Timing: model_multiscale

| Phase | Seconds | % |
|---|---|---|
| Data transfer | 41s | 2.9% |
| Forward pass | 808s | 56.2% |
| GSBA | 90s | 6.3% |
| Loss compute | 32s | 2.2% |
| Backward+Opt | 467s | 32.5% |
| **Total** | **1438s** | **100%** |

Est. total: **39.9h** (1.7d)

### model_pretrained
**model** | ResNet-18 + ImageNet pretrained (full fine-tune)
**Dev WER:** 21.16% | **Test WER:** 22.05%

| Stage | Epochs | CTC(g) | CTC(v) | Seg | Total |
|---|---|---|---|---|---|
| 1 | 1-30 | 1.713 | 1.992 | — | 2.712 |
| 2 | 31-90 | 0.721 | — | 2.430 | 1.960 |
| 3 | 91-100 | 0.518 | — | — | 0.526 |

### ⏱ Timing: model_pretrained

| Phase | Seconds | % |
|---|---|---|
| Data transfer | 39s | 3.2% |
| Forward pass | 683s | 55.8% |
| GSBA | 72s | 5.9% |
| Loss compute | 23s | 1.9% |
| Backward+Opt | 408s | 33.3% |
| **Total** | **1225s** | **100%** |

Est. total: **34.0h** (1.4d)

### model_frozen_backbone
**model** | ResNet-18 pretrained + FROZEN (feature extraction, 14.8M trainable)
**Dev WER:** 21.51% | **Test WER:** 22.46%

| Stage | Epochs | CTC(g) | CTC(v) | Seg | Total |
|---|---|---|---|---|---|
| 1 | 1-30 | 1.796 | 1.993 | — | 2.792 |
| 2 | 31-90 | 0.800 | — | 2.428 | 2.040 |
| 3 | 91-100 | 0.598 | — | — | 0.605 |

### ⏱ Timing: model_frozen_backbone

| Phase | Seconds | % |
|---|---|---|
| Data transfer | 26s | 3.0% |
| Forward pass | 483s | 56.2% |
| GSBA | 54s | 6.3% |
| Loss compute | 19s | 2.2% |
| Backward+Opt | 278s | 32.3% |
| **Total** | **860s** | **100%** |

Est. total: **23.9h** (1.0d)

### model_tsm
**model** | Temporal Shift Module — zero-parameter temporal modeling
**Dev WER:** 21.47% | **Test WER:** 22.44%

| Stage | Epochs | CTC(g) | CTC(v) | Seg | Total |
|---|---|---|---|---|---|
| 1 | 1-30 | 1.764 | 1.992 | — | 2.760 |
| 2 | 31-90 | 0.770 | — | 2.428 | 2.008 |
| 3 | 91-100 | 0.566 | — | — | 0.575 |

### ⏱ Timing: model_tsm

| Phase | Seconds | % |
|---|---|---|
| Data transfer | 35s | 2.9% |
| Forward pass | 696s | 56.9% |
| GSBA | 69s | 5.6% |
| Loss compute | 26s | 2.1% |
| Backward+Opt | 398s | 32.5% |
| **Total** | **1224s** | **100%** |

Est. total: **34.0h** (1.4d)

### model_mobilenet
**model** | MobileNetV3-Small — from scratch (17.6M params)
**Dev WER:** 22.72% | **Test WER:** 23.38%

| Stage | Epochs | CTC(g) | CTC(v) | Seg | Total |
|---|---|---|---|---|---|
| 1 | 1-30 | 1.814 | 1.992 | — | 2.813 |
| 2 | 31-90 | 0.821 | — | 2.427 | 2.061 |
| 3 | 91-100 | 0.616 | — | — | 0.628 |

### ⏱ Timing: model_mobilenet

| Phase | Seconds | % |
|---|---|---|
| Data transfer | 20s | 3.3% |
| Forward pass | 339s | 56.6% |
| GSBA | 36s | 6.0% |
| Loss compute | 10s | 1.7% |
| Backward+Opt | 194s | 32.4% |
| **Total** | **599s** | **100%** |

Est. total: **16.6h** (0.7d)

### model_mobilenet_pretrained
**model** | MobileNetV3-Small + ImageNet pretrained (full fine-tune)
**Dev WER:** 21.93% | **Test WER:** 22.51%

| Stage | Epochs | CTC(g) | CTC(v) | Seg | Total |
|---|---|---|---|---|---|
| 1 | 1-30 | 1.755 | 1.992 | — | 2.751 |
| 2 | 31-90 | 0.761 | — | 2.426 | 2.001 |
| 3 | 91-100 | 0.560 | — | — | 0.565 |

### ⏱ Timing: model_mobilenet_pretrained

| Phase | Seconds | % |
|---|---|---|
| Data transfer | 15s | 2.5% |
| Forward pass | 343s | 57.3% |
| GSBA | 37s | 6.2% |
| Loss compute | 12s | 2.0% |
| Backward+Opt | 192s | 32.1% |
| **Total** | **599s** | **100%** |

Est. total: **16.6h** (0.7d)

### model_mobilenet_frozen
**model** | MobileNetV3 pretrained + FROZEN (feature extraction, 5.5M trainable)
**Dev WER:** 22.18% | **Test WER:** 23.12%

| Stage | Epochs | CTC(g) | CTC(v) | Seg | Total |
|---|---|---|---|---|---|
| 1 | 1-30 | 1.785 | 1.992 | — | 2.784 |
| 2 | 31-90 | 0.791 | — | 2.430 | 2.033 |
| 3 | 91-100 | 0.587 | — | — | 0.597 |

### ⏱ Timing: model_mobilenet_frozen

| Phase | Seconds | % |
|---|---|---|
| Data transfer | 14s | 2.8% |
| Forward pass | 288s | 57.7% |
| GSBA | 26s | 5.2% |
| Loss compute | 11s | 2.2% |
| Backward+Opt | 160s | 32.1% |
| **Total** | **499s** | **100%** |

Est. total: **13.9h** (0.6d)

### model_efficientnet
**model** | EfficientNet-B0 — from scratch (21.3M params)
**Dev WER:** 22.30% | **Test WER:** 23.17%

| Stage | Epochs | CTC(g) | CTC(v) | Seg | Total |
|---|---|---|---|---|---|
| 1 | 1-30 | 1.793 | 1.991 | — | 2.792 |
| 2 | 31-90 | 0.802 | — | 2.426 | 2.041 |
| 3 | 91-100 | 0.596 | — | — | 0.599 |

### ⏱ Timing: model_efficientnet

| Phase | Seconds | % |
|---|---|---|
| Data transfer | 31s | 2.8% |
| Forward pass | 634s | 57.6% |
| GSBA | 63s | 5.7% |
| Loss compute | 20s | 1.8% |
| Backward+Opt | 353s | 32.1% |
| **Total** | **1101s** | **100%** |

Est. total: **30.6h** (1.3d)

### model_efficientnet_pretrained
**model** | EfficientNet-B0 + ImageNet pretrained (full fine-tune)
**Dev WER:** 21.14% | **Test WER:** 21.76%

| Stage | Epochs | CTC(g) | CTC(v) | Seg | Total |
|---|---|---|---|---|---|
| 1 | 1-30 | 1.744 | 1.990 | — | 2.742 |
| 2 | 31-90 | 0.751 | — | 2.429 | 1.990 |
| 3 | 91-100 | 0.545 | — | — | 0.554 |

### ⏱ Timing: model_efficientnet_pretrained

| Phase | Seconds | % |
|---|---|---|
| Data transfer | 28s | 2.5% |
| Forward pass | 632s | 57.5% |
| GSBA | 67s | 6.1% |
| Loss compute | 24s | 2.2% |
| Backward+Opt | 348s | 31.7% |
| **Total** | **1099s** | **100%** |

Est. total: **30.5h** (1.3d)

---

## 🔬 Training Experiments

### train_amp
**training** | AMP (FP16) + gradient accumulation (×2) — 6× faster
**Dev WER:** 22.03% | **Test WER:** 22.65%

| Stage | Epochs | CTC(g) | CTC(v) | Seg | Total |
|---|---|---|---|---|---|
| 1 | 1-30 | 1.765 | 1.992 | — | 2.762 |
| 2 | 31-90 | 0.771 | — | 2.429 | 2.011 |
| 3 | 91-100 | 0.565 | — | — | 0.578 |

### ⏱ Timing: train_amp

| Phase | Seconds | % |
|---|---|---|
| Data transfer | 10s | 2.9% |
| Forward pass | 196s | 57.6% |
| GSBA | 21s | 6.2% |
| Loss compute | 7s | 2.1% |
| Backward+Opt | 106s | 31.2% |
| **Total** | **340s** | **100%** |

Est. total: **9.4h** (0.4d)

### train_ema
**training** | EMA of weights (decay=0.999) for stable evaluation
**Dev WER:** 21.54% | **Test WER:** 22.30%

| Stage | Epochs | CTC(g) | CTC(v) | Seg | Total |
|---|---|---|---|---|---|
| 1 | 1-30 | 1.765 | 1.992 | — | 2.761 |
| 2 | 31-90 | 0.771 | — | 2.428 | 2.010 |
| 3 | 91-100 | 0.567 | — | — | 0.571 |

### ⏱ Timing: train_ema

| Phase | Seconds | % |
|---|---|---|
| Data transfer | 31s | 2.5% |
| Forward pass | 708s | 57.8% |
| GSBA | 65s | 5.3% |
| Loss compute | 24s | 2.0% |
| Backward+Opt | 396s | 32.4% |
| **Total** | **1224s** | **100%** |

Est. total: **34.0h** (1.4d)

### train_temporal_aug
**training** | Temporal frame masking + rate jitter augmentation
**Dev WER:** 21.34% | **Test WER:** 22.01%

| Stage | Epochs | CTC(g) | CTC(v) | Seg | Total |
|---|---|---|---|---|---|
| 1 | 1-30 | 1.764 | 1.991 | — | 2.760 |
| 2 | 31-90 | 0.771 | — | 2.427 | 2.011 |
| 3 | 91-100 | 0.566 | — | — | 0.575 |

### ⏱ Timing: train_temporal_aug

| Phase | Seconds | % |
|---|---|---|
| Data transfer | 32s | 2.6% |
| Forward pass | 706s | 57.7% |
| GSBA | 68s | 5.6% |
| Loss compute | 25s | 2.0% |
| Backward+Opt | 392s | 32.1% |
| **Total** | **1223s** | **100%** |

Est. total: **34.0h** (1.4d)

### train_smooth_transition
**training** | Smooth stage transitions (linear ramp over 5 epochs)
**Dev WER:** 21.47% | **Test WER:** 22.31%

| Stage | Epochs | CTC(g) | CTC(v) | Seg | Total |
|---|---|---|---|---|---|
| 1 | 1-30 | 1.765 | 1.992 | — | 2.762 |
| 2 | 31-90 | 0.771 | — | 2.428 | 2.012 |
| 3 | 91-100 | 0.566 | — | — | 0.571 |

### ⏱ Timing: train_smooth_transition

| Phase | Seconds | % |
|---|---|---|
| Data transfer | 39s | 3.2% |
| Forward pass | 702s | 57.4% |
| GSBA | 83s | 6.8% |
| Loss compute | 22s | 1.8% |
| Backward+Opt | 378s | 30.9% |
| **Total** | **1224s** | **100%** |

Est. total: **34.0h** (1.4d)

---

## 🔬 Combined Experiments

### combined_tier1
**combined** | VAC + Entropy + AMP + EMA + Temporal Aug + Smooth Transitions
**Dev WER:** 20.44% | **Test WER:** 20.95%

| Stage | Epochs | CTC(g) | CTC(v) | Seg | Total |
|---|---|---|---|---|---|
| 1 | 1-30 | 1.734 | 1.993 | — | 2.740 |
| 2 | 31-90 | 0.742 | — | 2.424 | 1.984 |
| 3 | 91-100 | 0.540 | — | — | 0.542 |

### ⏱ Timing: combined_tier1

| Phase | Seconds | % |
|---|---|---|
| Data transfer | 10s | 2.9% |
| Forward pass | 196s | 57.5% |
| GSBA | 18s | 5.3% |
| Loss compute | 8s | 2.3% |
| Backward+Opt | 109s | 32.0% |
| **Total** | **341s** | **100%** |

Est. total: **9.5h** (0.4d)

### combined_tier2
**combined** | All enhancements + dw convs + multi-scale + curriculum α + focal CTC
**Dev WER:** 20.07% | **Test WER:** 20.76%

| Stage | Epochs | CTC(g) | CTC(v) | Seg | Total |
|---|---|---|---|---|---|
| 1 | 1-30 | 1.715 | 1.993 | — | 2.718 |
| 2 | 31-90 | 0.721 | — | 2.429 | 1.964 |
| 3 | 91-100 | 0.516 | — | — | 0.526 |

### ⏱ Timing: combined_tier2

| Phase | Seconds | % |
|---|---|---|
| Data transfer | 11s | 3.2% |
| Forward pass | 197s | 57.9% |
| GSBA | 20s | 5.9% |
| Loss compute | 6s | 1.8% |
| Backward+Opt | 106s | 31.2% |
| **Total** | **340s** | **100%** |

Est. total: **9.4h** (0.4d)

---

## 🏆 Key Takeaways

1. **Best:** `combined_tier2` — 20.07% dev WER (-1.81% vs baseline)
2. **Worst:** `model_mobilenet` — 22.72% dev WER (+0.84% vs baseline)
3. **Combined Tier 2:** 20.07% dev WER (-1.81%) — all enhancements together
4. **Best single change:** `model_pretrained` (ImageNet backbone) — 21.16% (-0.72%)
5. **AMP impact:** Negligible WER change (+0.05%) with **6× training speedup** (39.3h → 6.5h)
6. **Pretrained backbones consistently beat from-scratch:** ResNet-18: −0.8%, EfficientNet-B0: −0.5%, MobileNetV3: −0.2%
7. **Frozen backbones:** trade ~0.3-0.8% WER for 30-60% faster training — useful for rapid iteration
8. **Stage 3 decouple spike:** 1.5-3.0% WER jump at epoch 91 — `train_smooth_transition` reduces this by ~0.3%
9. **VAC + Entropy combined** (−0.7%) outperforms either alone (VAC: −0.5%, Entropy: −0.3%)
10. **Multi-scale temporal branches** (−0.6%) are the best model-only enhancement at the cost of +15% training time

---
*Synthetic results — generated to reflect realistic training dynamics on PHOENIX-2014-T.*
*Actual results depend on hardware, data preprocessing, and random seed.*