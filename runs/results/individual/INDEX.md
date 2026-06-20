# SMKD — 7 Key Experiment Results

**Generated:** 2026-06-20 | **Dataset:** PHOENIX-2014-T | **GPU:** NVIDIA T4

---

## 📊 Comparison Table

| Experiment | Category | Dev WER | Test WER | Δ Baseline | Time |
|---|---|---|---|---|---|
| `baseline` | Reference | 21.88% | 22.56% | +0.00% | 34.0h |
| `loss_vac_entropy` | Loss Enhancement | 21.45% | 22.36% | -0.43% | 34.0h |
| `model_pretrained` | Model Enhancement | 20.96% | 21.48% | -0.92% | 34.0h |
| `model_transformer` | Model Enhancement | 21.31% | 21.86% | -0.57% | 27.2h |
| `train_amp` | Training Method | 21.87% | 22.51% | -0.00% | 9.4h |
| `combined_tier1` | Combined | 20.59% | 21.54% | -1.28% | 9.4h |
| `combined_tier2` | Combined | 19.88% | 20.67% | -2.00% | 9.4h |

---

## 🏆 Rankings

1. **Best overall:** `combined_tier2` (19.88%)
2. **Best single change:** `model_pretrained` (20.96%)
3. **Best loss enhancement:** `loss_vac_entropy` (21.45%)
4. **Transformer vs BiLSTM:** `model_transformer` (21.31%) — transformer is 80% of BiLSTM training time
5. **AMP speed:** `train_amp` (21.87%) — 3.6× faster

---

## 📁 Individual Reports

| Experiment | File |
|---|---|
| `baseline` | [`baseline.md`](baseline.md) |
| `loss_vac_entropy` | [`loss_vac_entropy.md`](loss_vac_entropy.md) |
| `model_pretrained` | [`model_pretrained.md`](model_pretrained.md) |
| `model_transformer` | [`model_transformer.md`](model_transformer.md) |
| `train_amp` | [`train_amp.md`](train_amp.md) |
| `combined_tier1` | [`combined_tier1.md`](combined_tier1.md) |
| `combined_tier2` | [`combined_tier2.md`](combined_tier2.md) |

---

*Synthetic results — generated to reflect realistic training dynamics.*
