#!/usr/bin/env python3
"""
SMKD Experiment Time Estimator for NVIDIA T4 GPU

Computes parameter counts, memory estimates, and training time estimates
for each experiment configuration.

Usage:
    python experiments/estimate.py
    python experiments/estimate.py --output estimates.csv
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================================
# Configuration
# ============================================================================

# T4 GPU specs
T4_FP32_TFLOPS = 8.1      # FP32 peak
T4_FP16_TFLOPS = 65.0     # FP16 Tensor Core peak
T4_MEMORY_GB = 16.0
T4_MEM_BW_GBPS = 320.0

# PHOENIX14-T dataset stats
PHOENIX_TRAIN_SAMPLES = 5672
PHOENIX_DEV_SAMPLES = 540
PHOENIX_AVG_FRAMES = 120      # average frames per sentence after subsampling
PHOENIX_MAX_FRAMES = 300
PHOENIX_VOCAB_SIZE = 1088     # ~1066 glosses + blank + unk
IMG_SIZE = 224

# Training config (default)
BATCH_SIZE = 2
EPOCHS = 100
EVAL_EVERY = 5

# Model component param counts (d_model=512, hidden_size=512)
# Verified against torch summary

COMPONENT_PARAMS = {
    # Spatial backbones (without FC head)
    "resnet18":        11_177_000,
    "mobilenet_v3_small": 2_543_000,
    "efficientnet_b0": 5_289_000,

    # Adapter to map backbone → d_model
    "adapter_512_512":         0,       # identity if bb_dim == d_model
    "adapter_576_512":     295_424,     # MobileNetV3 → 512
    "adapter_1280_512":    655_360,     # EfficientNet → 512

    # Temporal CNN (standard Conv1d)
    "tempcnn_standard":  2_623_488,     # 2×(512×512×5) + BN params

    # Temporal CNN (depthwise-separable)
    "tempcnn_dw":          531_456,     # 2×(512×5 + 512×512×1) + BN

    # Multi-scale extra branch (dilation=2)
    "tempcnn_ms_branch":   525_312,     # 1 extra branch (dwsep)

    # Multi-scale fusion (d_model*2 → d_model)
    "ms_fusion_2x":        525_312,     # 1×1 Conv1d(1024, 512) + BN
    "ms_fusion_3x":      1_050_624,     # 1×1 Conv1d(1536, 512) + BN

    # BiLSTM (2-layer, bidirectional, hidden=512)
    "bilstm":          10_502_144,      # LSTM + projection

    # NormalisedClassifier (vocab=1088)
    "classifier":         557_056,       # 1088 × 512

    # TSM overhead
    "tsm":                     0,        # zero-parameter
}


def model_params(backbone, conv_type, multi_scale, ms_dilations, vocab_size,
                 freeze_backbone=False):
    """Compute total and trainable parameter counts for a given model configuration."""
    bb = backbone
    total = 0
    frozen = 0

    # Backbone
    bb_params = COMPONENT_PARAMS[bb]
    total += bb_params
    if freeze_backbone:
        frozen += bb_params

    # Adapter
    if bb == "resnet18":
        adapter = COMPONENT_PARAMS["adapter_512_512"]
    elif bb == "mobilenet_v3_small":
        adapter = COMPONENT_PARAMS["adapter_576_512"]
    elif bb == "efficientnet_b0":
        adapter = COMPONENT_PARAMS["adapter_1280_512"]
    else:
        adapter = 0
    total += adapter

    # Temporal CNN
    if conv_type == "standard":
        total += COMPONENT_PARAMS["tempcnn_standard"]
    else:
        total += COMPONENT_PARAMS["tempcnn_dw"]

    # Multi-scale branches
    if multi_scale:
        n_branches = len(ms_dilations)
        for _ in range(n_branches):
            total += COMPONENT_PARAMS["tempcnn_ms_branch"]
        if n_branches == 1:
            total += COMPONENT_PARAMS["ms_fusion_2x"]
        elif n_branches == 2:
            total += COMPONENT_PARAMS["ms_fusion_3x"]

    # BiLSTM
    total += COMPONENT_PARAMS["bilstm"]

    # Classifiers (shared + 2 independent for stage 3)
    classifier_params = (vocab_size + 1) * 512
    total += 3 * classifier_params

    trainable = total - frozen
    return total, trainable, frozen


def estimate_flops_per_sample(backbone, conv_type, multi_scale, T_frames):
    """
    Approximate GFLOPs per training sample (forward + backward).
    
    Based on:
    - ResNet-18: ~1.8 GFLOPs per 224×224 image (forward)
    - MobileNetV3-Small: ~0.06 GFLOPs per image
    - EfficientNet-B0: ~0.39 GFLOPs per image
    - BiLSTM: ~0.002 GFLOPs per frame per direction
    - Backward ≈ 2× forward (for conv layers)
    """
    bb_gflops_per_frame = {
        "resnet18": 1.8,
        "mobilenet_v3_small": 0.06,
        "efficientnet_b0": 0.39,
    }

    frame_gflops = bb_gflops_per_frame.get(backbone, 1.8)

    # Spatial encoder: T frames * GFLOPs/frame
    spatial_fwd = T_frames * frame_gflops

    # Temporal CNN + BiLSTM: negligible compared to spatial (<1% of total)
    temporal_fwd = 0.02 * T_frames  # ~0.02 GFLOPs per frame

    # Forward total
    fwd_total = spatial_fwd + temporal_fwd

    # Training: forward + backward ≈ 3× forward
    train_total = fwd_total * 3.0

    # Multi-scale adds ~20% to temporal (still tiny)
    if multi_scale:
        train_total *= 1.02  # marginal overhead

    return train_total


def estimate_memory(backbone, batch_size, T_frames, img_size, use_amp, use_ema):
    """
    Estimate GPU memory usage in GB.
    
    Key memory consumers:
    1. Model parameters + gradients + optimizer states
    2. Activations (proportional to batch_size × T × spatial_resolution)
    3. AMP overhead (if enabled, ~40% less activation memory)
    4. EMA model copy (if enabled, ~params × 4 bytes extra)
    """
    params, _, _ = model_params(backbone, "standard", False, [], PHOENIX_VOCAB_SIZE)

    # 1. Parameters + gradients + Adam states (2 momentum buffers)
    param_mem = params * 4 / (1024**3)       # float32 params
    grad_mem = params * 4 / (1024**3)        # gradients
    optim_mem = params * 8 / (1024**3)       # Adam: 2 × float32 per param
    static_mem = param_mem + grad_mem + optim_mem

    # 2. Activations (dominated by ResNet feature maps)
    # Per-frame activation: ~15-20 MB for ResNet-18 at 224×224
    # With batch_size=2, T=120: 2 * 120 * 18MB ≈ 4.3 GB for ResNet-18
    # (rough heuristic based on ResNet-18 intermediate feature maps)
    if backbone == "resnet18":
        per_frame_act_mb = 18
    elif backbone == "mobilenet_v3_small":
        per_frame_act_mb = 5
    else:  # efficientnet_b0
        per_frame_act_mb = 12

    activation_mem = batch_size * T_frames * per_frame_act_mb / 1024  # GB

    # AMP reduces activation memory ~40%
    if use_amp:
        activation_mem *= 0.60

    # 3. EMA model (separate param copy)
    ema_mem = (params * 4 / (1024**3)) if use_ema else 0

    total = static_mem + activation_mem + ema_mem + 0.5  # 0.5 GB overhead
    return total


def estimate_epoch_time(backbone, batch_size, T_frames, use_amp, n_gpus,
                        multi_scale=False, extra_overhead=1.0):
    """
    Estimate per-epoch training time in minutes.

    Methodology:
    - Compute FLOPs per sample
    - Divide by GPU throughput (accounting for real-world efficiency ~50%)
    - Account for DataLoader, loss computation, GSBA overhead
    - Add BiLSTM sequential bottleneck floor
    """
    n_samples = PHOENIX_TRAIN_SAMPLES
    n_batches = n_samples // batch_size

    gflops_per_sample = estimate_flops_per_sample(backbone, "standard", multi_scale, T_frames)
    total_gflops = n_samples * gflops_per_sample  # per epoch

    if use_amp:
        gpu_tflops = T4_FP16_TFLOPS * 0.35  # ~35% real efficiency on mixed workloads
    else:
        gpu_tflops = T4_FP32_TFLOPS * 0.45  # ~45% efficiency on conv-heavy workloads

    compute_seconds = total_gflops / (gpu_tflops * 1000)

    # BiLSTM sequential bottleneck: ~800-1500 timesteps/sec on T4 for this LSTM size
    # With T=120 per sample, 5672 samples, batch=2 → ~340K timesteps
    # At 1000 timesteps/sec: 340 sec minimum
    lstm_timesteps = (n_samples * T_frames) / batch_size  # effective
    lstm_throughput = 1200  # timesteps/sec on T4 for 2×512 BiLSTM
    lstm_seconds = lstm_timesteps / lstm_throughput

    # Take max of FLOP-based and LSTM-bottleneck estimates
    compute_seconds = max(compute_seconds, lstm_seconds)

    # Data loading, GSBA, CTC loss overhead: ~20%
    compute_seconds *= 1.20

    # Extra overhead factor (from enhancements)
    compute_seconds *= extra_overhead

    # Multi-GPU scaling (DataParallel ≈ 0.85× perfect scaling)
    if n_gpus > 1:
        compute_seconds /= (n_gpus * 0.85)

    return compute_seconds / 60  # minutes


def estimate_eval_time(batch_size, T_frames, n_gpus):
    """Estimate per-evaluation time in minutes."""
    n_samples = PHOENIX_DEV_SAMPLES
    n_batches = n_samples // batch_size

    # Eval is ~1/3 of training time per sample (no backward)
    gflops_per_sample = estimate_flops_per_sample(
        "resnet18", "standard", False, T_frames
    ) / 3.0

    total_gflops = n_samples * gflops_per_sample
    gpu_tflops = T4_FP32_TFLOPS * 0.50
    seconds = total_gflops / (gpu_tflops * 1000) * 1.2  # +20% overhead

    if n_gpus > 1:
        seconds /= (n_gpus * 0.85)

    return seconds / 60


# ============================================================================
# Experiment definitions
# ============================================================================

EXPERIMENTS = [
    # ---- Loss experiments ----
    {
        "name": "baseline",
        "config": "baseline.yaml",
        "category": "loss",
        "backbone": "resnet18",
        "conv_type": "standard",
        "multi_scale": False,
        "use_amp": False,
        "use_ema": False,
        "freeze_backbone": False,
        "extra_overhead": 1.00,
    },
    {
        "name": "loss_vac",
        "config": "loss_vac.yaml",
        "category": "loss",
        "backbone": "resnet18",
        "conv_type": "standard",
        "multi_scale": False,
        "use_amp": False,
        "use_ema": False,
        "freeze_backbone": False,
        "extra_overhead": 1.03,
    },
    {
        "name": "loss_entropy",
        "config": "loss_entropy.yaml",
        "category": "loss",
        "backbone": "resnet18",
        "conv_type": "standard",
        "multi_scale": False,
        "use_amp": False,
        "use_ema": False,
        "freeze_backbone": False,
        "extra_overhead": 1.02,
    },
    {
        "name": "loss_vac_entropy",
        "config": "loss_vac_entropy.yaml",
        "category": "loss",
        "backbone": "resnet18",
        "conv_type": "standard",
        "multi_scale": False,
        "use_amp": False,
        "use_ema": False,
        "freeze_backbone": False,
        "extra_overhead": 1.05,
    },
    {
        "name": "loss_confidence_gsba",
        "config": "loss_confidence_gsba.yaml",
        "category": "loss",
        "backbone": "resnet18",
        "conv_type": "standard",
        "multi_scale": False,
        "use_amp": False,
        "use_ema": False,
        "freeze_backbone": False,
        "extra_overhead": 1.05,
    },
    {
        "name": "loss_curriculum_alpha",
        "config": "loss_curriculum_alpha.yaml",
        "category": "loss",
        "backbone": "resnet18",
        "conv_type": "standard",
        "multi_scale": False,
        "use_amp": False,
        "use_ema": False,
        "freeze_backbone": False,
        "extra_overhead": 1.00,
    },
    {
        "name": "loss_focal_ctc",
        "config": "loss_focal_ctc.yaml",
        "category": "loss",
        "backbone": "resnet18",
        "conv_type": "standard",
        "multi_scale": False,
        "use_amp": False,
        "use_ema": False,
        "freeze_backbone": False,
        "extra_overhead": 1.01,
    },

    # ---- Model experiments ----
    {
        "name": "model_dwconv",
        "config": "model_dwconv.yaml",
        "category": "model",
        "backbone": "resnet18",
        "conv_type": "depthwise_separable",
        "multi_scale": False,
        "use_amp": False,
        "use_ema": False,
        "freeze_backbone": False,
        "extra_overhead": 0.98,
    },
    {
        "name": "model_multiscale",
        "config": "model_multiscale.yaml",
        "category": "model",
        "backbone": "resnet18",
        "conv_type": "standard",
        "multi_scale": True,
        "use_amp": False,
        "use_ema": False,
        "freeze_backbone": False,
        "extra_overhead": 1.15,
    },
    {
        "name": "model_mobilenet",
        "config": "model_mobilenet.yaml",
        "category": "model",
        "backbone": "mobilenet_v3_small",
        "conv_type": "standard",
        "multi_scale": False,
        "use_amp": False,
        "use_ema": False,
        "freeze_backbone": False,
        "extra_overhead": 0.30,
    },
    {
        "name": "model_tsm",
        "config": "model_tsm.yaml",
        "category": "model",
        "backbone": "resnet18",
        "conv_type": "standard",
        "multi_scale": False,
        "use_amp": False,
        "use_ema": False,
        "freeze_backbone": False,
        "extra_overhead": 1.02,
    },
    {
        "name": "model_pretrained",
        "config": "model_pretrained.yaml",
        "category": "model",
        "backbone": "resnet18",
        "conv_type": "standard",
        "multi_scale": False,
        "use_amp": False,
        "use_ema": False,
        "freeze_backbone": False,  # pretrained weights, full fine-tuning
        "extra_overhead": 1.00,
    },
    {
        "name": "model_frozen_backbone",
        "config": "model_frozen_backbone.yaml",
        "category": "model",
        "backbone": "resnet18",
        "conv_type": "standard",
        "multi_scale": False,
        "use_amp": False,
        "use_ema": False,
        "freeze_backbone": True,   # pretrained + frozen (feature extraction)
        "extra_overhead": 0.70,    # ~30% faster: no backbone backward pass
    },

    # ---- Training experiments ----
    {
        "name": "train_amp",
        "config": "train_amp.yaml",
        "category": "training",
        "backbone": "resnet18",
        "conv_type": "standard",
        "multi_scale": False,
        "use_amp": True,
        "use_ema": False,
        "freeze_backbone": False,
        "extra_overhead": 1.00,
    },
    {
        "name": "train_ema",
        "config": "train_ema.yaml",
        "category": "training",
        "backbone": "resnet18",
        "conv_type": "standard",
        "multi_scale": False,
        "use_amp": False,
        "use_ema": True,
        "freeze_backbone": False,
        "extra_overhead": 1.02,
    },
    {
        "name": "train_temporal_aug",
        "config": "train_temporal_aug.yaml",
        "category": "training",
        "backbone": "resnet18",
        "conv_type": "standard",
        "multi_scale": False,
        "use_amp": False,
        "use_ema": False,
        "freeze_backbone": False,
        "extra_overhead": 1.01,
    },
    {
        "name": "train_smooth_transition",
        "config": "train_smooth_transition.yaml",
        "category": "training",
        "backbone": "resnet18",
        "conv_type": "standard",
        "multi_scale": False,
        "use_amp": False,
        "use_ema": False,
        "freeze_backbone": False,
        "extra_overhead": 1.00,
    },

    # ---- Combined ----
    {
        "name": "combined_tier1",
        "config": "combined_tier1.yaml",
        "category": "combined",
        "backbone": "resnet18",
        "conv_type": "standard",
        "multi_scale": False,
        "use_amp": True,
        "use_ema": True,
        "freeze_backbone": False,
        "extra_overhead": 1.08,
    },
    {
        "name": "combined_tier2",
        "config": "combined_tier2.yaml",
        "category": "combined",
        "backbone": "resnet18",
        "conv_type": "depthwise_separable",
        "multi_scale": True,
        "use_amp": True,
        "use_ema": True,
        "freeze_backbone": False,
        "extra_overhead": 1.22,
    },
]


# ============================================================================
# Main
# ============================================================================

def main():
    import argparse
    p = argparse.ArgumentParser(description="SMKD T4 training time estimator")
    p.add_argument("--output", "-o", default=None, help="Save estimates to CSV")
    p.add_argument("--n-gpus", type=int, default=1, help="Number of T4 GPUs")
    p.add_argument("--batch-size", type=int, default=2, help="Per-GPU batch size")
    p.add_argument("--short", action="store_true", help="Short output (table only)")
    args = p.parse_args()

    T_frames = PHOENIX_AVG_FRAMES
    n_gpus = args.n_gpus
    batch_size = args.batch_size

    if not args.short:
        print("=" * 110)
        print("  SMKD Experiment Time Estimator — NVIDIA T4 GPU")
        print("=" * 110)
        print(f"  GPU          : {n_gpus}× T4 (FP32: {T4_FP32_TFLOPS} TFLOPS, FP16: {T4_FP16_TFLOPS} TFLOPS, {T4_MEMORY_GB} GB)")
        print(f"  Dataset      : PHOENIX14-T ({PHOENIX_TRAIN_SAMPLES} train, {PHOENIX_DEV_SAMPLES} dev, ~{PHOENIX_VOCAB_SIZE} classes)")
        print(f"  Avg frames   : {T_frames} per sentence")
        print(f"  Image size   : {IMG_SIZE}×{IMG_SIZE}")
        print(f"  Batch size   : {batch_size} (per GPU)")
        print(f"  Epochs       : {EPOCHS}")
        print(f"  Eval every   : {EVAL_EVERY} epochs")
        print()

    # Compute estimates
    rows = []
    for exp in EXPERIMENTS:
        bb = exp["backbone"]
        ct = exp["conv_type"]
        ms = exp["multi_scale"]
        amp = exp["use_amp"]
        ema = exp["use_ema"]

        params, trainable, frozen = model_params(
            bb, ct, ms, [1, 2], PHOENIX_VOCAB_SIZE, exp["freeze_backbone"])
        mem = estimate_memory(bb, batch_size, T_frames, IMG_SIZE, amp, ema)
        epoch_min = estimate_epoch_time(bb, batch_size, T_frames, amp, n_gpus,
                                        ms, exp["extra_overhead"])
        eval_min = estimate_eval_time(batch_size, T_frames, n_gpus)
        n_evals = EPOCHS // EVAL_EVERY + 1  # +1 for final eval
        total_hours = (epoch_min * EPOCHS + eval_min * n_evals) / 60

        rows.append({
            "name": exp["name"],
            "config": exp["config"],
            "category": exp["category"],
            "backbone": bb,
            "params_m": params / 1e6,
            "trainable_m": trainable / 1e6,
            "frozen_m": frozen / 1e6,
            "mem_gb": mem,
            "mem_status": "OK" if mem < T4_MEMORY_GB * 0.9 else "⚠ OOM",
            "epoch_min": epoch_min,
            "total_hours": total_hours,
            "total_days": total_hours / 24,
            "use_amp": amp,
            "use_ema": ema,
        })

    # Print table
    header = f"{'Experiment':<24} {'Backbone':<20} {'Total':>7} {'Train':>7} {'Frozen':>7} {'Memory':>8} {'Ep(min)':>8} {'Tot(h)':>8} {'Tot(d)':>8} {'AMP':>4} {'EMA':>4}"
    sep = "-" * len(header)

    print(sep)
    print(header)
    print(sep)

    for r in rows:
        mem_str = f"{r['mem_gb']:.1f} GB"
        if "OOM" in r["mem_status"]:
            mem_str += " ⚠"
        frozen_str = f"{r['frozen_m']:.1f}M" if r["frozen_m"] > 0 else "     -"
        print(f"{r['name']:<24} {r['backbone']:<20} {r['params_m']:6.1f}M {r['trainable_m']:6.1f}M {frozen_str:>7} {mem_str:>12} {r['epoch_min']:7.1f}m {r['total_hours']:7.1f}h {r['total_days']:7.1f}d {'✓' if r['use_amp'] else '':>4} {'✓' if r['use_ema'] else '':>4}")

    print(sep)

    # Category summary
    if not args.short:
        print()
        print("─" * 80)
        print("  Summary by Category")
        print("─" * 80)
        from collections import defaultdict
        cats = defaultdict(list)
        for r in rows:
            cats[r["category"]].append(r)

        for cat, cat_rows in sorted(cats.items()):
            total_h = sum(r["total_hours"] for r in cat_rows)
            print(f"  {cat:<12}: {len(cat_rows)} experiments, {total_h:.1f} total GPU-hours ({total_h/24:.1f} GPU-days)")
            for r in cat_rows:
                print(f"    - {r['name']:<24} {r['total_hours']:6.1f}h")

        grand_total = sum(r["total_hours"] for r in rows)
        print(f"\n  {'ALL':<12}: {len(rows)} experiments, {grand_total:.1f} total GPU-hours ({grand_total/24:.1f} GPU-days)")

        # Practical recommendation
        print()
        print("─" * 80)
        print("  Practical Notes")
        print("─" * 80)
        print(f"  • Baseline (1×T4, no AMP):     ~{rows[0]['total_days']:.1f} days")
        print(f"  • Baseline (1×T4, AMP):         ~{rows[10]['total_days']:.1f} days")
        print(f"  • All 16 experiments (1×T4):    ~{grand_total/24:.1f} GPU-days")
        print(f"  • All 16 experiments (2×T4):    ~{grand_total/24/1.7:.1f} GPU-days (DataParallel)")
        print(f"  • Recommended: run Tier 1 first ({sum(r['total_hours'] for r in rows if r['category'] in ('loss', 'training')):.1f}h),")
        print(f"    then model experiments separately")
        print(f"  • Synthetic data smoke test: ~30s per experiment, negligible")
        print()

    # CSV output
    if args.output:
        import csv
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"Estimates saved to: {args.output}")


if __name__ == "__main__":
    main()
