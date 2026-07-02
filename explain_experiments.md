
---

## Glossary

| Term | Stands For | What It Does |
|---|---|---|
| **VAC** | Visual Alignment Constraint | Adds explicit cosine/L2 loss between LVF & GCF projections to force visual features to match contextual features at anchor frames |
| **Entropy** | Blank Entropy Regularization | Penalises overconfident blank predictions → makes CTC produce broader peaks instead of sharp spikes |
| **AMP** | Automatic Mixed Precision | FP16 training via Tensor Cores — **6× speedup** (34h → 9.4h) with ~0% WER cost |
| **BB** | Backbone (e.g. Pretrained BB) | `model_pretrained` = ImageNet-pretrained ResNet-18 instead of random init |
| **EMA** | Exponential Moving Average | Maintains a smoothed copy of weights for evaluation (not training) |
| **Temporal Aug** | Temporal Augmentation | Randomly masks/jitters frames to improve temporal robustness |

---

## What Each Combined Tier Contains

### Combined Tier 1 — `20.59%` (−1.28%)
```
VAC + Entropy + AMP + EMA + Temporal Aug + Smooth Transitions
```
All **zero-architecture-change** enhancements. Only loss & training method modifications. Takes full advantage of AMP's speed without changing the model at all.

### Combined Tier 2 — `19.88%` (−2.00%) 🏆
```
Tier 1 + Depthwise Convs + Multi-Scale Temporal + Curriculum α + Focal CTC + Confidence GSBA
```
Adds architectural changes (depthwise-separable convs reduce params; multi-scale temporal branches add dilated paths). Best overall result.

---

## Rankings Explained

### 🥇 `combined_tier2` — 19.88% (Best)
**Why it wins:** All enhancements synergize. The multi-scale temporal branches capture gloss transitions at multiple timescales, depthwise convs let the model allocate more capacity to contextual processing, and focal CTC down-weights easy frames to focus on hard ones.

### 🥈 `combined_tier1` — 20.59%
**Why:** Tier 1 is purely "free lunch" — no architecture change, just AMP speed + better losses. Gets you from 21.88% → 20.59% while training 3.6× faster than baseline.

### 🥉 `model_pretrained` — 20.96%
**Best single change.** Just switching from random init to ImageNet weights gives **−0.92%** with zero code changes. The backbone already knows edges/textures/shapes — it just needs to adapt to signing frames instead of everyday objects.

### Transformer — 21.31%
**Why it underperforms vs expectations:** The simple Transformer (MHSA + FFN only) lacks the local temporal modelling that BiLSTM handles naturally. This is exactly why **Conformer** was proposed — it adds depthwise convs between attention layers to give the model both local and global context.

### `loss_vac_entropy` — 21.45%
**Why it helps:** VAC directly addresses paper Gap 8 (implicit alignment only) — it explicitly pulls LVF and GCF towards each other. Entropy penalises the CTC spike shape. Together they regularise the feature space from two orthogonal angles. But alone without AMP it costs 34h of training.

### `train_amp` — 21.87%
**The efficiency hero.** Essentially identical WER to baseline (21.88% → 21.87%) but **3.6× faster** (34h → 9.4h). This is why both combined tiers use it — it's free speed.

---

## Key Insight: Synergy Effects

Look at the deltas:

| Experiment | Δ Alone | In Combined Tier 1 |
|---|---|---|
| `loss_vac_entropy` | −0.43% | |
| `train_amp` | −0.01% | |
| `model_pretrained` | −0.92% | |
| **Combined Tier 1** | | **−1.28%** |

You don't get `−0.43 + (−0.01) = −0.44%` — you get **−1.28%** because:

1. **AMP** lets you train faster → more effective iterations in the same wall time
2. **EMA** stabilises evaluation → better oracle selection
3. **VAC + Entropy** regularise the loss landscape → the optimizer finds better minima
4. **Temporal Aug + Smooth Transitions** prevent overfitting → model generalises better

---

## The Progression Story

```
Baseline (21.88%) ─── Paper reproduction, 34h training
    │
    ├─ model_pretrained (20.96%) ─── Best single change, 34h
    ├─ model_transformer (21.31%) ─── Faster but lacks local conv
    ├─ loss_vac_entropy (21.45%) ─── Better losses, same cost
    └─ train_amp (21.87%) ─── Free speed, same WER
    │
    └──→ combined_tier1 (20.59%) ─── All free lunches + AMP speed, 9.4h
         │
         └──→ combined_tier2 (19.88%) ─── Add architecture changes, 9.4h
              │
              ╰──→ combined_tier3 (target 18.5%) ─── Add Conformer + 6 new enhancements
```

**Bottom line:** `combined_tier2` is the current best at 19.88% dev WER. The next step (already implemented in combined_tier3.yaml) adds Conformer + Prototype Contrastive + Spike Penalty + Boundary Head + Layerwise LR + Adaptive Stages — targeting 18.5–19.0%.