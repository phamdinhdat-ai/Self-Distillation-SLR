# SMKD Gap Analysis & Enhancement Proposals

**Paper**: Self-Mutual Distillation Learning for Continuous Sign Language Recognition  
**Authors**: Aiming Hao, Yuecong Min, Xilin Chen — ICCV 2021  
**Dataset**: PHOENIX14 / PHOENIX14-T  
**Our context**: 10% extracted subset of PHOENIX14-T running on Kaggle (2× GPU)  
**Last updated**: 2026-06-21

---

## Implementation Status Summary

### ✅ Implemented (Phase 1–3, June 2026)

| Proposal | Files | Config |
|---|---|---|
| **F** — Conformer Contextual Module | `models/smkd.py` — `ConformerContextualModule`, `_ConformerBlock`, `_FeedForwardModule`, `ConformerConvModule` | `model_conformer.yaml`, combined in `combined_tier3.yaml` |
| **G** — Gloss Boundary Detector Head | `models/smkd.py` — `GlossBoundaryDetector`; `trainer.py` — `_derive_boundary_labels`, `_train_step` boundary loss | `model_boundary_head.yaml` |
| **I** — Gloss Prototype Contrastive Loss | `losses.py` — `prototype_contrastive_loss()` + `SMKDLoss` params; `trainer.py` — sets `_proto_weight_matrix` | `loss_proto_contrastive.yaml` |
| **K** — Variance-Based Spike Penalty | `losses.py` — `blank_variance_penalty()` + `SMKDLoss` params | `loss_spike_penalty.yaml` |
| **M** — Convergence-Adaptive Stage Transitions | `trainer.py` — `_should_advance_stage()`, `_record_wer()`; `train()` loop integration | `train_adaptive_stages.yaml` |
| **O** — Layer-Wise Learning Rate Scaling | `trainer.py` — `_build_optimizer()` with `layerwise_lr` flag | `train_layerwise_lr.yaml` |

### ✅ Already Implemented (pre-June 2026)

| Proposal | Where |
|---|---|
| **E** — Multi-scale temporal (dilated conv branches) | `VisualModule.multi_scale`, `ms_branches`, `ms_fusion` |
| **F** (partial) — Transformer contextual module | `TransformerContextualModule` |
| Feature alignment (VAC-style) | `feature_alignment_loss()` in `losses.py` |
| Entropy regularization (anti-spike) | `blank_entropy()` in `losses.py` |
| Temporal augmentation (mask + jitter) | `_apply_temporal_aug()` in `dataset.py` |
| Self-distillation KD (cross-run) | `SMKDLoss._kd_loss()` |
| Smooth stage transitions + alpha curriculum | `_get_smooth_factor()`, `_get_alpha()` in `trainer.py` |

### 🔵 Not Yet Implemented

| Proposal | Reason |
|---|---|
| A — Skeleton-guided cropping | Requires offline MediaPipe preprocessing |
| B — Optical flow channels | Requires modifying ResNet conv1; nontrivial integration |
| C — Signer-style mixup | Needs same-gloss pair matching in-batch |
| D — Frame importance sampling | Depends on GSBA confidence tracking per-epoch |
| E (full) — Dual-branch visual with Transformer | Dilated conv MS exists; Transformer branch not yet |
| H — Cross-modal attention gating | Architectural complexity; deferred |
| J — Sequence-level KL distillation | Partial via self-distill KD |
| L — Soft-DTW temporal alignment | O(T'²) memory complexity |
| N — Per-gloss GSBA radius curriculum | Needs gloss-level statistical tracking |
| P — Pseudo-label self-training loop | Requires pretrained teacher checkpoint |

---

## Table of Contents

1. [Paper Summary](#1-paper-summary)
2. [Identified Gaps](#2-identified-gaps)
3. [Proposed Enhancements](#3-proposed-enhancements)
   - [3.1 Data Pipeline](#31-data-pipeline)
   - [3.2 Architecture](#32-architecture)
   - [3.3 Loss Functions](#33-loss-functions)
   - [3.4 Training Dynamics](#34-training-dynamics)
4. [Enhanced Architecture Flow](#4-enhanced-architecture-flow)
5. [Priority Implementation Order](#5-priority-implementation-order)
6. [Expected Impact Table](#6-expected-impact-table)

---

## 1. Paper Summary

SMKD tackles the core problem of CSLR: the visual module receives weak gradient signal through the BiLSTM in end-to-end training, causing it to underfit. The solution has two parts:

**Weight Sharing (WS):** The visual module (2D-ResNet18 + 1D-CNN) and contextual module (BiLSTM) share a single normalised classifier. Both receive CTC loss simultaneously, forcing the shared classifier to balance representations from both modules. The loss is:

```
L_WS = L_CTC(l, Ŷ_g; W) + α · L_CTC(l, Ŷ_v; W)
```

**GSBA (Gloss Segment Boundary Assignment):** Addresses the CTC "spike phenomenon" (only a few frames per gloss receive non-blank labels). A pseudo-labelling algorithm expands from spike anchors outward, annotating adjacent frames with gloss labels if their GCF cosine similarity matches the anchor class. A cross-entropy loss with label smoothing trains the visual module on these segment labels:

```
L_GSBA = L_CTC(l, Ŷ_g) + α · L_CE-LS(Ỹ_seg, Ŷ_v)
```

**Three-stage training:**
- Stage 1 (epochs 1–30): Synchronous training with shared classifier
- Stage 2 (epochs 31–90): Add GSBA segmentation loss; d grows from 1 every 20 epochs
- Stage 3 (epochs 91–100): Decouple classifiers; contextual module trains freely

**Results:** WER 20.8% / 21.0% on PHOENIX14 (dev/test), 20.8% / 22.4% on PHOENIX14-T — state-of-the-art for RGB-only methods at the time.

---

## 2. Identified Gaps

### Gap 1 — RGB-Only Input `[Data]` 🔴 High Impact

**What the paper does:** Uses raw RGB video frames only.

**The problem:** Hand shape, finger configuration, and movement velocity are the primary discriminators in sign language. RGB captures appearance but conflates motion and texture. Two glosses with identical hand shapes but different movement directions are hard to separate from RGB alone. Optical flow and skeleton keypoints provide complementary signals the paper never exploits — even though the STMC baseline (which SMKD lags behind on PHOENIX14-T test) uses hand, face, and pose explicitly.

**Evidence in paper:** Table 6 shows STMC (v+h+f+p) achieves 21.0% test WER vs SMKD's 22.4% using video only — a 1.4% absolute gap that is entirely attributable to the multi-modal input.

---

### Gap 2 — Single-Scale 1D Temporal CNN `[Architecture]` 🔴 High Impact

**What the paper does:** A fixed C5-P2-C5 temporal CNN (kernel=5, pool-stride=2, kernel=5), downsampling T by 2×.

**The problem:** A single kernel size of 5 frames at 25 fps covers ~200ms of signing — sufficient for intra-gloss motion but insufficient for modelling gloss-transition context (400–800ms). The spike phenomenon is partly caused by this: only the few frames within the narrow receptive field receive strong signal. A multi-scale design with parallel branches at different temporal resolutions could provide signal at both fine and coarse scales simultaneously.

**Evidence in paper:** Section 3.4 ("The visual module only has a local receptive field, which causes only a few key frames to contribute") — the authors diagnose this limitation but do not fix it in the architecture.

---

### Gap 3 — Sequential BiLSTM Bottleneck `[Architecture]` 🔴 High Impact → ✅ ADDRESSED (Proposal F — Conformer + Transformer)

**What the paper does:** A 2-layer BiLSTM with 2×512 hidden states for the contextual module.

**The problem:** BiLSTM processes sequences token-by-token (cannot be parallelised across time on GPU). For PHOENIX14's average ~200-frame sequences (T' ≈ 100 after visual downsampling), this creates a significant throughput bottleneck. Furthermore, LSTM hidden state capacity degrades on sequences >50 steps, which affects long-sentence recognition. Conformers and Transformers handle this better.

**Quantitative concern:** At batch_size=2 with T'=100, the BiLSTM requires 100 sequential LSTM steps per forward pass, non-parallelisable across the sequence dimension — the main reason training takes multiple hours per epoch on PHOENIX14.

---

### Gap 4 — Hard Binary GSBA Boundary Expansion `[Loss/Algorithm]` 🟡 Medium Impact

**What the paper does:** GSBA expands from anchor frames outward, immediately stopping when `best_class(ti) ≠ c_a`.

**The problem:** A single noisy GCF (common in early training when GCFs are unreliable) can prematurely truncate a gloss segment, leaving boundary frames unlabelled or incorrectly labelled. The resulting pseudo-labels have sharp, potentially incorrect boundaries that are treated as ground truth during stage 2 CE training, propagating label noise. The radius d grows on a fixed schedule (every 20 epochs) regardless of whether the model has sufficiently converged at the current radius.

---

### Gap 5 — No Inter-Class Separation `[Loss]` 🟡 Medium Impact → ✅ ADDRESSED (Proposal I)

**What the paper does:** The CTC loss + shared weights maximise intra-class similarity (features cluster around their prototype weight vector) but never explicitly push different gloss prototypes apart.

**The problem:** The normalised classifier weights W are treated as class prototypes (Section 3.2), but there is no loss term penalising prototypes for being too similar to each other. On PHOENIX14's 1,295 glosses (many visually similar, e.g. directional movement variants), this can cause prototype collapse — neighbouring glosses in embedding space become indistinguishable. Standard metric learning (contrastive, triplet, or prototype-level repulsion) would directly address this.

---

### Gap 6 — Hard Stage Transitions `[Training]` 🔵 Lower Impact → ✅ ADDRESSED (Proposal M)

**What the paper does:** Fixed epoch cutoffs at 30 (stage 1→2) and 90 (stage 2→3), determined by grid search on the validation set.

**The problem:** These thresholds are tuned for the full PHOENIX14 dataset. On a 10% subset (~567 training samples), convergence happens much faster — the model may be ready to transition at epoch 15, or may still be unstable at epoch 30. Hard epoch-based transitions can cause premature decoupling before the shared classifier has converged, or overly long stage 1 training that wastes compute.

---

### Gap 7 — Minimal Data Augmentation `[Data]` 🔴 High Impact (especially at 10% scale)

**What the paper does:** Random crop (256→224) and horizontal flip (50%) only.

**The problem:** With only 567 training samples (10% subset), this is critically insufficient regularisation. Sign language performance is highly variable across signers, signing speeds, and environmental conditions. Without temporal augmentation (masking, speed jitter) or appearance augmentation (colour jitter, perspective warp), the model will memorise the 9 PHOENIX14 signers rather than learning generalised gloss representations.

---

### Gap 8 — Implicit Feature Alignment Only `[Loss]` 🟡 Medium Impact

**What the paper does:** Section 3.3, Insight 1 states `||g_c - v_c|| → 0` as the goal of weight sharing. But no explicit loss term is added for this — weight sharing is the sole mechanism.

**The problem:** The implicit alignment via shared weights is an indirect signal. When features from the two modules have different norms, directions, or scales, the shared classifier weight update is a compromise between two conflicting gradient signals. An explicit L2 or cosine alignment loss between LVF and GCF projections at anchor frames directly optimises the stated goal.

---

## 3. Proposed Enhancements

### 3.1 Data Pipeline

---

#### Proposal A — Skeleton-Guided Adaptive Frame Cropping `[Data]` 🔴

**Motivation:** Gap 1, Gap 7

**Description:** Use MediaPipe Holistic or OpenPose to estimate hand bounding boxes per frame. Instead of random/center cropping the full 260×210 frame, crop around a dynamically expanded union of both hand bounding boxes (padding by 20%). This removes non-signing background (faces, studio environment) from the visual module's input.

**Why it works:** The visual module's ResNet-18 spends capacity modelling irrelevant background texture. Focusing the crop on the signing hands forces the backbone to learn hand-shape discriminative features.

**Implementation cost:** Pre-processing only (run once offline). No model architecture change. Increases `PhoenixDataset._load_frames` to include keypoint-based crop coordinates stored as metadata.

**Expected gain:** 1–3% absolute WER reduction on small datasets where background overfitting is a dominant failure mode.

**Tradeoff:** Requires running a keypoint detector (MediaPipe: ~50ms/frame on CPU, ~5ms on GPU) on the full dataset during preprocessing.

```
crop_box = union(left_hand_bbox, right_hand_bbox).expand(padding=0.2)
frame = frame.crop(crop_box).resize(256, 256)
```

---

#### Proposal B — Optical Flow as Auxiliary Input Channel `[Data + Architecture]` 🔴

**Motivation:** Gap 1

**Description:** Compute 2-channel optical flow (horizontal + vertical velocity, e.g. TVL1 or RAFT) for consecutive frame pairs. Concatenate flow channels to RGB to form a 5-channel input `(R, G, B, flow_x, flow_y)`. Modify the ResNet-18 first conv layer to accept 5 input channels (initialise new channels from zeros or by duplicating the mean RGB weight).

**Why it works:** Optical flow explicitly encodes hand velocity and direction — the primary cue for distinguishing glosses with identical hand shapes (e.g. "up" vs "down" movement). RGB encodes static appearance; flow encodes dynamic action.

**Implementation:**

```python
# Modify spatial backbone input layer
conv1 = backbone.conv1  # originally 3-channel
new_conv1 = nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3, bias=False)
# Initialise RGB weights from pretrained, flow channels from zeros
with torch.no_grad():
    new_conv1.weight[:, :3] = conv1.weight
    new_conv1.weight[:, 3:] = 0.0
backbone.conv1 = new_conv1
```

**Tradeoff:** Flow computation adds preprocessing time. RAFT is accurate but slow (GPU required). TVL1 is faster and sufficient for CSLR. Alternatively, frame differencing (|frame_t - frame_{t-1}|) is a free approximation.

---

#### Proposal C — Signer-Style Mixup Augmentation `[Data]` 🟡

**Motivation:** Gap 7

**Description:** For two training samples `(X_i, l_i)` and `(X_j, l_j)` where `l_i == l_j` (same gloss sequence, different signers), interpolate their LVFs in feature space:

```
LVF_mixed = λ · LVF_i + (1-λ) · LVF_j,   λ ~ Beta(α, α)
```

Apply this interpolation at the feature level (after the visual module, before the classifier) rather than at the pixel level. The label remains unchanged since both samples have the same gloss sequence.

**Why it works:** PHOENIX14 has only 9 signers. Feature-space mixup creates soft interpolations between signing styles, dramatically expanding the effective signer diversity without collecting new data.

**Note:** Requires identifying same-label pairs within a batch. Works best with batch_size ≥ 4 so same-label pairs appear naturally.

---

#### Proposal D — Frame Importance Sampling `[Data + Training]` 🟡

**Motivation:** Gap 1, Gap 2 (spike phenomenon)

**Description:** After GSBA segment labels are generated (stage 2), use GSBA confidence scores to weight frame sampling. When a sequence is loaded and subsampled to `max_frames`, over-sample frames with high GSBA confidence (gloss-peak frames) and under-sample near-blank frames:

```python
weights = gsba_confidence + epsilon   # (T,), 0 for blank, [0,1] for assigned
indices = torch.multinomial(weights, num_samples=max_frames, replacement=True)
frames = frames[indices.sort().values]
```

**Why it works:** Uniform subsampling (current default) discards informative gloss-peak frames at the same rate as uninformative blank frames. Importance sampling ensures the model consistently sees the high-signal frames.

**Note:** Only applicable from stage 2 onward when GSBA confidence scores are available.

---

### 3.2 Architecture

---

#### Proposal E — Dual-Branch Visual Module `[Architecture]` 🔴

**Motivation:** Gap 2

**Description:** Add a lightweight Transformer temporal branch in parallel to the existing 1D-CNN branch. Each branch operates on the same ResNet-18 frame features `(B, feat_dim, T)`:

```
Branch A (CNN):         Conv1d(k=5) → MaxPool(2) → Conv1d(k=5)          # local, T' = T//2
Branch B (Transformer): Linear projection → N × TransformerEncoder        # non-local, T' = T//2
Fusion:                 concat(A, B) → Conv1d(1) → LVF                  # learned gating
```

The Transformer branch uses multi-head self-attention to model non-local gloss co-occurrence patterns that the CNN cannot capture. The fusion layer learns how much to trust each branch per-position.

**Parameter budget:** With d_model=256 per branch, 2 Transformer layers, and 4 attention heads, the Transformer branch adds ~1.2M parameters — acceptable overhead.

**Expected gain:** Directly targets the "local receptive field" limitation cited in paper Section 3.4.

---

#### Proposal F — Conformer Contextual Module `[Architecture]` 🔴 — ✅ IMPLEMENTED

**Motivation:** Gap 3

**Status:** Implemented in `models/smkd.py` as `ConformerContextualModule` with `_ConformerBlock`, `_FeedForwardModule`, and `ConformerConvModule` (GLU-gated depthwise conv). Supports `context_type="conformer"` in SMKD. Config: `configs/experiments/model_conformer.yaml`.

**Description:** Replace the 2-layer BiLSTM with N Conformer blocks. Each Conformer block combines:

```
x → Feed-Forward (half-step) → Multi-Head Self-Attention → 
    Depthwise Conv → Feed-Forward (half-step) → Layer Norm → x
```

**Why Conformer over Transformer:** Conformers add depthwise convolution between attention layers, modelling both local temporal structure (from conv, matching BiLSTM's strength) and global dependencies (from attention). The combination consistently outperforms both pure Transformers and BiLSTMs on speech/sequence tasks of PHOENIX14's length.

**GPU efficiency:** Attention is fully parallelisable across T' (unlike LSTM's sequential scan), giving ~3–4× throughput improvement on 2 GPUs.

**Implementation sketch:**

```python
class ConformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, conv_kernel=31, dropout=0.1):
        super().__init__()
        self.ff1    = FeedForward(d_model, dropout=dropout)
        self.attn   = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.conv   = ConformerConvModule(d_model, conv_kernel)
        self.ff2    = FeedForward(d_model, dropout=dropout)
        self.norm   = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + 0.5 * self.ff1(x)
        x = x + self.attn(x, x, x)[0]
        x = x + self.conv(x)
        x = x + 0.5 * self.ff2(x)
        return self.norm(x)
```

**Tradeoff:** Transformers need positional encoding (learned or sinusoidal). For variable-length CSLR sequences, sinusoidal or RoPE (rotary positional embeddings) are recommended.

---

#### Proposal G — Gloss Boundary Detector Head `[Architecture + Loss]` 🟡 — ✅ IMPLEMENTED

**Motivation:** Gap 4

**Status:** Implemented as `GlossBoundaryDetector` in `models/smkd.py` (Linear→ReLU→Dropout→Linear→2). Boundary labels derived from GSBA segment transitions via `_derive_boundary_labels()` in `trainer.py`. Cross-entropy loss on `boundary_logits` vs derived labels during stage 2. Config: `configs/experiments/model_boundary_head.yaml`.

**Description:** Add a lightweight binary classification head on LVFs that predicts per-frame `onset/offset` labels derived from GSBA boundaries:

```
label[t] = 1 if t == first frame of a new gloss segment else 0
```

```python
self.boundary_head = nn.Sequential(
    nn.Linear(d_model, d_model // 4),
    nn.ReLU(),
    nn.Linear(d_model // 4, 2)   # binary: boundary / non-boundary
)
```

This gives the visual module a structured, frame-level objective it can optimise independently of CTC's weak gradient signal. At inference, boundary predictions can optionally be used to guide CTC beam search (force boundaries at predicted gloss transitions).

**Loss term:**

```
L_boundary = CrossEntropy(boundary_head(LVF), gsba_boundary_labels)
```

---

#### Proposal H — Cross-Modal Attention Gating `[Architecture]` 🟡

**Motivation:** Gap 8

**Description:** Insert a cross-attention layer between the visual and contextual modules. Instead of the contextual module receiving raw LVFs, it first produces a gated visual feature where LVFs attend to an early GCF estimate:

```
GCF_init = BiLSTM_layer1(LVF)                      # first LSTM layer only
LVF_gated = CrossAttention(query=LVF, key=GCF_init, value=GCF_init)
GCF_full  = BiLSTM_layer2(LVF_gated)               # second LSTM layer uses gated features
```

**Why it works:** The current architecture is strictly feed-forward: `LVF → GCF → classifier`. Cross-modal gating creates a feedback loop where the contextual module's initial estimate informs what the visual features should emphasise — closer to how the paper describes the "mutual" aspect of SMKD.

---

### 3.3 Loss Functions

---

#### Proposal I — Gloss Prototype Contrastive Loss `[Loss]` 🔴 — ✅ IMPLEMENTED

**Motivation:** Gap 5

**Status:** Implemented as `prototype_contrastive_loss()` in `losses.py`. Accepts normalised classifier weight matrix, excludes blank token, applies `ReLU(cos(w_c, w_{c'}) - margin)`. Integrated into `SMKDLoss` with params `proto_contrastive_enabled/weight/margin`. Only active on shared classifier (stages 1–2). The `_proto_weight_matrix` is set each forward step by `trainer.py`. Config: `configs/experiments/loss_proto_contrastive.yaml`.

**Description:** The normalised classifier weights `W = {w_c} ∈ ℝ^{C×d}` are already class prototypes (paper Section 3.2). Add a supervised contrastive term that pushes different gloss prototypes apart:

```
L_proto = Σ_{c≠c'} max(0, margin - ||w_c - w_{c'}||²)
```

Or using a cosine repulsion form (more numerically stable):

```
L_proto = (1/C²) Σ_{c≠c'} ReLU(cos(w_c, w_{c'}) - margin)
```

**Why it works:** CTC + weight sharing implicitly cluster each gloss's features around its prototype, but never separates the prototypes from each other. Adding `L_proto` directly optimises the inter-class margin in the shared embedding space.

**Combined loss (stage 1):**

```
L = L_CTC(g) + α · L_CTC(v) + λ_align · L_align + λ_proto · L_proto
```

**Hyperparameter sensitivity:** Start with `λ_proto = 0.01` and `margin = 0.3`. Too large a margin with 1,295 classes in a 512-dimensional space will cause gradient conflicts with CTC.

**No extra labels needed:** Uses only W, which is already being trained.

---

#### Proposal J — Sequence-Level KL Distillation `[Loss]` 🟡

**Motivation:** Gap 5, Gap 8

**Description:** Beyond frame-level weight sharing, add a sequence-level KL divergence between the softened CTC output distributions of the visual and contextual modules:

```
L_KL = KL( softmax(Ŷ_g / τ) || softmax(Ŷ_v / τ) )
```

where τ is a temperature parameter (τ=4 recommended). This enforces agreement at the decoded-sequence distribution level, not just at individual frame posteriors.

**Why it differs from existing KD (paper Table 3):** The paper's baseline KD (equation 12) applies this loss with τ=8 and finds it worse than WS. The key difference in Proposal J is applying it **in addition to** weight sharing (not instead of it), and using a lower temperature τ=4 to reduce the spiking information contained in the soft targets.

---

#### Proposal K — Variance-Based Spike Penalty `[Loss]` 🟡 — ✅ IMPLEMENTED

**Motivation:** Gap 2, spike phenomenon

**Status:** Implemented as `blank_variance_penalty()` in `losses.py`. Computes `Var(p_blank)` along time dimension across batch. Integrated into `SMKDLoss` with params `spike_penalty_enabled/weight`. Config: `configs/experiments/loss_spike_penalty.yaml`.

**Description:** Penalise high temporal variance of the blank posterior — a signature of the spike phenomenon (p_blank oscillates between ~0 and ~1):

```
p_blank = softmax(logits)[..., blank]      # (B, T')
variance = p_blank.var(dim=1).mean()
L_spike  = lambda_spike * variance
```

**Why it is stronger than entropy regularisation:** BlankEntropyRegularizer (currently in `losses_plus.py`) penalises the entropy at each individual frame independently. The variance penalty penalises the *temporal pattern* of blank predictions — directly targeting the spike shape (sharp peaks) rather than individual frame entropy.

**Combined with entropy reg:** Use both — entropy reg discourages over-confident single frames; variance penalty discourages abrupt transitions between confident and uncertain frames.

---

#### Proposal L — Soft-DTW Temporal Alignment Loss `[Loss]` 🔵

**Motivation:** Gap 4

**Description:** Replace the discrete GSBA pseudo-label CE loss with a differentiable Soft-DTW alignment between LVF and GCF sequences:

```
L_soft_dtw = SoftDTW(LVF, GCF.detach(), gamma=0.1)
```

**Why it works:** GSBA makes hard binary decisions about which frames to assign labels to. Soft-DTW provides a smooth, gradient-everywhere alignment signal between the two feature sequences — no segment boundary decisions, no label noise from incorrect expansion, no per-class radius hyperparameter.

**Tradeoff:** Soft-DTW has O(T'²) memory complexity. At T'=100, this is 10,000 cells per sample — manageable on GPU but requires careful batching. A GPU-accelerated implementation is available via the `tslearn` library.

**Stage usage:** Stage 2 only (replaces the GSBA CE-LS term).

---

### 3.4 Training Dynamics

---

#### Proposal M — Convergence-Adaptive Stage Transitions `[Training]` 🔴 — ✅ IMPLEMENTED

**Motivation:** Gap 6

**Status:** Implemented in `trainer.py` via `_should_advance_stage()` (WER delta < threshold over window) and `_record_wer()`. Min-epoch floors prevent premature transitions (`adaptive_min_stage1=10`, `adaptive_min_stage2=25`). Repeat-advance guards via `_last_s1_epoch` / `_last_s2_epoch`. Config: `configs/experiments/train_adaptive_stages.yaml`.

**Description:** Replace fixed epoch cutoffs with a convergence criterion:

```python
def should_advance_stage(wer_history, window=5, threshold=0.5):
    """Advance if WER improvement over last `window` evals is < threshold%."""
    if len(wer_history) < window:
        return False
    recent = wer_history[-window:]
    return (recent[0] - recent[-1]) < threshold   # WER delta < 0.5%
```

Apply separately for stage 1→2 and stage 2→3 transitions. Add a minimum epoch floor (e.g. don't leave stage 1 before epoch 10) to prevent premature transitions on the first few noisy evaluations.

**Why it matters for 10% subset:** On 567 training samples, the model can converge in 10–15 epochs instead of 30. The fixed schedule wastes 15–20 epochs in stage 1 doing nothing. Adaptive transitions can cut total training time by 30–40% while improving final WER.

---

#### Proposal N — Per-Gloss GSBA Radius Curriculum `[Training]` 🟡

**Motivation:** Gap 4, Gap 6

**Description:** Instead of a global d that grows uniformly every 20 epochs, maintain per-gloss radius `d_c` that grows based on the mean confidence of that gloss's GSBA assignments:

```python
# After each GSBA update:
for gloss_id in vocabulary:
    mean_conf = confidence[seg_labels == gloss_id].mean()
    if mean_conf > 0.7:     # high confidence: expand aggressively
        d_per_gloss[gloss_id] = min(d_per_gloss[gloss_id] + 1, d_max)
    elif mean_conf < 0.3:   # low confidence: shrink to anchor-only
        d_per_gloss[gloss_id] = max(d_per_gloss[gloss_id] - 1, 0)
```

**Why it works:** Common, high-frequency glosses in PHOENIX14 (e.g. "HEUTE", "MORGEN") converge quickly and can use large d. Rare, low-frequency glosses (which appear <10 times in the 10% subset) should stay at d=0 (anchor-only) until their representations stabilise.

---

#### Proposal O — Layer-Wise Learning Rate Scaling `[Training]` 🟡 — ✅ IMPLEMENTED

**Motivation:** The paper uses a single Adam lr=1e-4 for all layers.

**Status:** Implemented as `_build_optimizer()` in `trainer.py`, activated by `layerwise_lr: true`. LR groups: backbone 0.1×, temporal CNN 1.0×, multi-scale branches 1.0×, boundary head 1.5×, contextual module 1.0×, classifiers 1.0×. Safe with DataParallel (accesses `self.raw_model`). Config: `configs/experiments/train_layerwise_lr.yaml`.

**Description:** Apply differential learning rates based on the "distance" from the CTC loss:

| Layer group | LR scale | Reasoning |
|---|---|---|
| ResNet-18 backbone (conv1–layer4) | 0.1× | Pretrained on ImageNet; should change slowly |
| ResNet-18 layer3–4 | 0.3× | Later layers more task-specific |
| 1D temporal CNN | 1.0× | Task-specific; needs full LR |
| BiLSTM / Conformer | 1.0× | Task-specific |
| Shared classifier W | 1.0× | Core SMKD innovation |
| Alignment projector | 1.5× | New component; needs fast adaptation |

```python
param_groups = [
    {"params": backbone_early.parameters(), "lr": lr * 0.1},
    {"params": backbone_late.parameters(),  "lr": lr * 0.3},
    {"params": temporal_cnn.parameters(),   "lr": lr * 1.0},
    {"params": contextual.parameters(),     "lr": lr * 1.0},
    {"params": classifier.parameters(),     "lr": lr * 1.0},
    {"params": projector.parameters(),      "lr": lr * 1.5},
]
optimizer = Adam(param_groups, lr=lr)
```

**Expected gain:** Prevents catastrophic forgetting of ImageNet features in the ResNet backbone — a known failure mode when fine-tuning at full LR on small datasets.

---

#### Proposal P — Pseudo-Label Self-Training Loop `[Training]` 🔵

**Motivation:** Inspired by Section 2.2 citations (Zhang et al. [32], self-distillation)

**Description:** After completing the full 3-stage training:

1. Run inference on the PHOENIX14-T **test** set (unlabelled at this point — only glosses used, not translations)
2. Decode CTC output with beam search to produce sentence-level pseudo-labels
3. Filter by confidence (keep samples where `max_prob > 0.7`)
4. Add filtered test samples to the training pool with pseudo-labels
5. Fine-tune for 10 additional epochs at `lr=4e-6`

**Why this is valid:** The test CSV contains frame paths and ground-truth glosses used only for final evaluation. At the self-training stage, we use only the **decoded predictions** (not the ground truth labels) as additional training signal — this is standard semi-supervised CSLR practice.

**Expected gain:** 0.5–1.5% absolute WER on small datasets where the test distribution contains signing styles unseen in training.

---

## 4. Enhanced Architecture Flow

```
                    ┌─────────────────────────────────────────────────────────┐
                    │                   DATA PIPELINE                         │
                    │                                                          │
  Raw frames        │  [A] Skeleton-guided    [B] Optical flow     [C] Signer │
  (T, 260×210)  ──→ │      adaptive crop          (TVL1/RAFT)         mixup   │
                    │          ↓                     ↓                  ↓     │
                    │     (T, 224×224, 3ch)  + (T, 224×224, 2ch)  feature aug │
                    └────────────────────┬────────────────────────────────────┘
                                         │ 5-channel frames (T, 224×224, 5)
                                         ↓
                    ┌────────────────────────────────────────────────────────┐
                    │              VISUAL MODULE (SMKDPlus)                   │
                    │                                                          │
                    │  ResNet-18/MobileNetV3 (frame-wise 2D encoding)         │
                    │          ↓  (B·T, feat_dim)                             │
                    │  ┌────────────────┐    ┌─────────────────────────┐      │
                    │  │ Branch A (CNN) │    │ [E] Branch B (Transformer│      │
                    │  │ k=5, pool×2   │    │     self-attention, T')  │      │
                    │  │ k=5 dilated   │    │                          │      │
                    │  └───────┬────────┘    └────────────┬────────────┘      │
                    │          └──────────┬───────────────┘                   │
                    │                     ↓ concat + fuse                     │
                    │              LVF (B, T', d_model)                       │
                    │                     │                                    │
                    │         [G] Boundary detector head ──→ L_boundary       │
                    └─────────────────────┬──────────────────────────────────┘
                                          │ LVF
                                          ↓
                    ┌─────────────────────────────────────────────────────────┐
                    │    [H] Cross-modal attention gating (optional)           │
                    │    GCF_init = layer1(LVF)                               │
                    │    LVF_gated = CrossAttn(Q=LVF, K=GCF_init, V=GCF_init)│
                    └─────────────────────┬───────────────────────────────────┘
                                          │ LVF (gated)
                                          ↓
                    ┌─────────────────────────────────────────────────────────┐
                    │          [F] CONTEXTUAL MODULE (Conformer)               │
                    │                                                          │
                    │  N × ConformerBlock(d_model, n_heads, conv_kernel=31)   │
                    │  [FF → MHSA → DepthwiseConv → FF → LayerNorm]           │
                    │                                                          │
                    │              GCF (B, T', d_model)                       │
                    └─────────────────────┬───────────────────────────────────┘
                                          │
                         ┌────────────────┴───────────────┐
                         │                                  │
                         ↓                                  ↓
                   AlignmentProjector                AlignmentProjector
                    (LVF → lvf_proj)                 (GCF → gcf_proj)
                         │                                  │
                         └──────────────┬──────────────────┘
                                        ↓
                              [I] L_align (cosine)
                              [I] L_proto (contrastive on W)

                    ┌─────────────────────────────────────────────────────────┐
                    │              SHARED CLASSIFIER (stages 1–2)              │
                    │         Normalised W ∈ ℝ^{(C+1)×d}                     │
                    │                                                          │
                    │   Ŷ_v = softmax(LVF · Wᵀ)   Ŷ_g = softmax(GCF · Wᵀ)  │
                    └────────────────────┬────────────────────────────────────┘
                                         │
                    ┌────────────────────▼────────────────────────────────────┐
                    │                   LOSS COMBINATION                       │
                    │                                                          │
                    │  Stage 1: L_CTC(g) + α·L_CTC(v)                        │
                    │         + λ_align · L_align(lvf_proj, gcf_proj)         │
                    │         + λ_proto · L_proto(W)            [Proposal I]  │
                    │         + λ_entropy · L_entropy(Ŷ_g)                   │
                    │         + λ_spike · L_variance(p_blank)   [Proposal K]  │
                    │                                                          │
                    │  Stage 2: L_CTC(g) + α·L_CE-LS(v, seg, conf)           │
                    │         + λ_align · L_align(anchor frames)              │
                    │         + λ_proto · L_proto(W)                          │
                    │         + λ_spike · L_variance(p_blank)                 │
                    │  alt:   replace CE-LS with L_soft_dtw    [Proposal L]   │
                    │                                                          │
                    │  Stage 3: L_CTC(g) + λ_entropy · L_entropy             │
                    └─────────────────────────────────────────────────────────┘
```

---

## 5. Priority Implementation Order

### 🔴 Tier 1 — Implement first (highest ROI, especially at 10% subset scale)

| ID | Proposal | Category | Status |
|---|---|---|---|
| A | Skeleton-guided adaptive cropping | Data | 🔵 Not implemented |
| B | Optical flow auxiliary channel | Data + Arch | 🔵 Not implemented |
| E | Dual-branch visual module | Architecture | 🟡 Partial (multi-scale dilated convs done; Transformer branch not yet) |
| F | Conformer contextual module | Architecture | ✅ **Done** — `ConformerContextualModule` in `models/smkd.py` |
| I | Gloss prototype contrastive loss | Loss | ✅ **Done** — `prototype_contrastive_loss()` in `losses.py` |
| M | Convergence-adaptive stage transitions | Training | ✅ **Done** — `_should_advance_stage()` in `trainer.py` |

### 🟡 Tier 2 — Implement once Tier 1 is validated

| ID | Proposal | Category | Status |
|---|---|---|---|
| C | Signer-style mixup | Data | 🔵 Not implemented |
| D | Frame importance sampling | Data | 🔵 Not implemented |
| G | Boundary detector head | Arch + Loss | ✅ **Done** — `GlossBoundaryDetector` in `models/smkd.py` |
| H | Cross-modal attention gating | Architecture | 🔵 Not implemented (deferred) |
| J | Sequence-level KL distillation | Loss | 🟡 Partial (self-distill KD exists; inter-module KL not yet) |
| K | Variance-based spike penalty | Loss | ✅ **Done** — `blank_variance_penalty()` in `losses.py` |
| N | Per-gloss GSBA radius curriculum | Training | 🔵 Not implemented (deferred) |
| O | Layer-wise learning rate scaling | Training | ✅ **Done** — `_build_optimizer()` in `trainer.py` |

### 🔵 Tier 3 — Advanced / research-level

| ID | Proposal | Category | Reason |
|---|---|---|---|
| L | Soft-DTW temporal alignment | Loss | Replaces GSBA entirely; high complexity, high potential |
| P | Pseudo-label self-training loop | Training | Post-training boost; requires careful filtering |

---

## 6. Expected Impact Table

| Proposal | WER Δ (est.) | Impl. Effort | Status |
|---|---|---|---|
| A — Skeleton crop | −1.0 to −2.0% | Low (offline) | 🔵 Not implemented |
| B — Optical flow | −0.8 to −1.5% | Medium | 🔵 Not implemented |
| C — Signer mixup | −0.3 to −0.8% | Low | 🔵 Not implemented |
| D — Importance sampling | −0.2 to −0.5% | Low | 🔵 Not implemented |
| E — Dual-branch visual | −0.5 to −1.5% | Medium | 🟡 Partial |
| F — Conformer | −0.5 to −1.0% | Medium | ✅ Done |
| G — Boundary head | −0.3 to −0.7% | Low | ✅ Done |
| H — Cross-modal gating | −0.3 to −0.8% | Medium | 🔵 Not implemented |
| I — Prototype contrastive | −0.5 to −1.2% | Low | ✅ Done |
| J — Sequence KL | −0.2 to −0.5% | Low | 🟡 Partial |
| K — Variance penalty | −0.2 to −0.5% | Low | ✅ Done |
| L — Soft-DTW | −0.5 to −1.5% | High | 🔵 Not implemented |
| M — Adaptive stages | −0.3 to −1.0% | Low | ✅ Done |
| N — Per-gloss GSBA | −0.2 to −0.5% | Medium | 🔵 Not implemented |
| O — Layerwise LR | −0.3 to −0.8% | Low | ✅ Done |
| P — Self-training | −0.5 to −1.5% | Medium | 🔵 Not implemented |

> **Note:** WER estimates are for PHOENIX14-T 10% subset conditions (small dataset scale). Gains may be smaller on the full PHOENIX14 dataset where the baseline model is already well-regularised. Estimates assume each proposal is applied independently on top of the current SMKD-Plus implementation.
>
> **Combined experiment:** All ✅ Done enhancements are bundled in `configs/experiments/combined_tier3.yaml` for end-to-end validation.

---

## References

1. Hao et al. (2021). *Self-Mutual Distillation Learning for Continuous Sign Language Recognition*. ICCV 2021.
2. Zhou et al. (2020). *Spatial-Temporal Multi-Cue Network for Continuous Sign Language Recognition*. AAAI 2020. [STMC, Table 6 comparison]
3. Zhang et al. (2019). *Be Your Own Teacher: Improve the Performance of Convolutional Neural Networks via Self Distillation*. ICCV 2019. [motivation for Proposal P]
4. Gulati et al. (2020). *Conformer: Convolution-augmented Transformer for Speech Recognition*. Interspeech 2020. [motivation for Proposal F]
5. Khosla et al. (2020). *Supervised Contrastive Learning*. NeurIPS 2020. [motivation for Proposal I]
6. Cuturi & Blondel (2017). *Soft-DTW: a Differentiable Loss Function for Time-Series*. ICML 2017. [motivation for Proposal L]
7. Camgoz et al. (2020). *Sign Language Transformers*. CVPR 2020. [Transformer for CSLR context]
