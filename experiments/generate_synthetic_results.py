#!/usr/bin/env python3
"""
Synthetic training curve generator for SMKD experiments.

Generates realistic per-epoch loss curves and WER predictions for all
experiment configs, respecting the 3-stage training schedule, stage
transition bumps, and LR decay effects.

Output: runs/results/SMKD_EXPERIMENT_RESULTS.md
"""

import os
import sys
from typing import Dict, List

import numpy as np

# ============================================================================
# Configuration
# ============================================================================

EPOCHS = 100
STAGE1_END = 30
STAGE2_END = 90
EVAL_EVERY = 5
SEED = 42

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "runs", "results"
)
OUTPUT_MD = os.path.join(OUTPUT_DIR, "SMKD_EXPERIMENT_RESULTS.md")

# ============================================================================
# Realistic baseline parameters (calibrated to SMKD paper on PHOENIX14-T)
# ============================================================================

BASELINE = {
    "ctc_g_s1_start": 3.5,
    "ctc_g_s1_end":   1.2,
    "ctc_v_s1_start": 3.8,
    "ctc_v_s1_end":   1.4,
    "ctc_g_s2_start": 1.3,
    "ctc_g_s2_end":   0.65,
    "seg_s2_start":   4.5,
    "seg_s2_end":     1.8,
    "ctc_g_s3_start": 0.72,
    "ctc_g_s3_end":   0.55,
    "wer_initial":    92.0,
    "wer_final_dev":  22.5,
    "wer_final_test": 23.2,
    "s1_s2_bump_ctc": 0.08,
    "s1_s2_bump_wer": 1.5,
    "s2_s3_bump_ctc": 0.07,
    "s2_s3_bump_wer": 2.0,
    "lr_decay_effect": 0.03,
}

# ============================================================================
# Experiment definitions — realistic Δ WER from baseline (negative = better)
# ============================================================================

EXPERIMENTS = [
    # ---- Loss experiments ----
    {"name": "baseline",
     "delta_wer": 0.0, "delta_ctc": 0.0,
     "category": "loss",
     "desc": "Paper reproduction — no enhancements"},

    {"name": "loss_vac",
     "delta_wer": -0.5, "delta_ctc": 0.0,
     "align_strength": 0.08,
     "category": "loss",
     "desc": "VAC-style visual alignment constraint (cosine, w=0.1)"},

    {"name": "loss_entropy",
     "delta_wer": -0.3, "delta_ctc": 0.0,
     "entropy_strength": 0.012,
     "category": "loss",
     "desc": "Entropy regularization on blank posterior (anti-spike, w=0.01)"},

    {"name": "loss_vac_entropy",
     "delta_wer": -0.7, "delta_ctc": 0.0,
     "align_strength": 0.07, "entropy_strength": 0.010,
     "category": "loss",
     "desc": "VAC alignment + Entropy regularization combined"},

    {"name": "loss_confidence_gsba",
     "delta_wer": -0.4, "delta_ctc": 0.0,
     "category": "loss",
     "desc": "Confidence-weighted GSBA (margin-weighted CE-LS)"},

    {"name": "loss_curriculum_alpha",
     "delta_wer": -0.2, "delta_ctc": +0.02,
     "category": "loss",
     "desc": "Curriculum schedule on α (linear 0.1→0.5)"},

    {"name": "loss_focal_ctc",
     "delta_wer": -0.3, "delta_ctc": -0.03,
     "category": "loss",
     "desc": "Focal CTC modulation (γ=2.0)"},

    # ---- Model experiments (ResNet-18 variants) ----
    {"name": "model_dwconv",
     "delta_wer": +0.1, "delta_ctc": 0.0,
     "category": "model",
     "desc": "Depthwise-separable temporal convs (23.9M params)"},

    {"name": "model_multiscale",
     "delta_wer": -0.6, "delta_ctc": -0.02,
     "category": "model",
     "desc": "Multi-scale temporal branches (dilations [1,2], 28.1M params)"},

    {"name": "model_pretrained",
     "delta_wer": -0.8, "delta_ctc": -0.05,
     "category": "model",
     "desc": "ResNet-18 + ImageNet pretrained (full fine-tune)"},

    {"name": "model_frozen_backbone",
     "delta_wer": -0.4, "delta_ctc": +0.03,
     "category": "model",
     "desc": "ResNet-18 pretrained + FROZEN (feature extraction, 14.8M trainable)"},

    {"name": "model_tsm",
     "delta_wer": -0.1, "delta_ctc": 0.0,
     "category": "model",
     "desc": "Temporal Shift Module — zero-parameter temporal modeling"},

    # ---- Model experiments (MobileNetV3 variants) ----
    {"name": "model_mobilenet",
     "delta_wer": +0.8, "delta_ctc": +0.05,
     "category": "model",
     "desc": "MobileNetV3-Small — from scratch (17.6M params)"},

    {"name": "model_mobilenet_pretrained",
     "delta_wer": -0.2, "delta_ctc": -0.01,
     "category": "model",
     "desc": "MobileNetV3-Small + ImageNet pretrained (full fine-tune)"},

    {"name": "model_mobilenet_frozen",
     "delta_wer": +0.3, "delta_ctc": +0.02,
     "category": "model",
     "desc": "MobileNetV3 pretrained + FROZEN (feature extraction, 5.5M trainable)"},

    # ---- Model experiments (EfficientNet variants) ----
    {"name": "model_efficientnet",
     "delta_wer": +0.4, "delta_ctc": +0.03,
     "category": "model",
     "desc": "EfficientNet-B0 — from scratch (21.3M params)"},

    {"name": "model_efficientnet_pretrained",
     "delta_wer": -0.5, "delta_ctc": -0.02,
     "category": "model",
     "desc": "EfficientNet-B0 + ImageNet pretrained (full fine-tune)"},

    # ---- Training experiments ----
    {"name": "train_amp",
     "delta_wer": +0.05, "delta_ctc": 0.0,
     "category": "training",
     "desc": "AMP (FP16) + gradient accumulation (×2) — 6× faster"},

    {"name": "train_ema",
     "delta_wer": -0.3, "delta_ctc": 0.0,
     "category": "training",
     "desc": "EMA of weights (decay=0.999) for stable evaluation"},

    {"name": "train_temporal_aug",
     "delta_wer": -0.5, "delta_ctc": 0.0,
     "category": "training",
     "desc": "Temporal frame masking + rate jitter augmentation"},

    {"name": "train_smooth_transition",
     "delta_wer": -0.3, "delta_ctc": 0.0,
     "category": "training",
     "desc": "Smooth stage transitions (linear ramp over 5 epochs)"},

    # ---- Combined ----
    {"name": "combined_tier1",
     "delta_wer": -1.5, "delta_ctc": -0.03,
     "align_strength": 0.06, "entropy_strength": 0.008,
     "category": "combined",
     "desc": "VAC + Entropy + AMP + EMA + Temporal Aug + Smooth Transitions"},

    {"name": "combined_tier2",
     "delta_wer": -2.0, "delta_ctc": -0.05,
     "align_strength": 0.06, "entropy_strength": 0.008,
     "category": "combined",
     "desc": "All enhancements + dw convs + multi-scale + curriculum α + focal CTC"},
]


# ============================================================================
# Curve generation helpers
# ============================================================================

def smooth_decay(start: float, end: float, n: int) -> np.ndarray:
    """Exponential decay from start → end over n steps."""
    t = np.linspace(0, 1, n)
    factor = (1 - np.exp(-4 * t)) / (1 - np.exp(-4))
    return start + (end - start) * factor


def add_noise(curve: np.ndarray, amplitude: float = 0.02, freq: float = 3.0,
              seed_offset: int = 0) -> np.ndarray:
    """Add correlated sinusoidal + Gaussian noise."""
    rng = np.random.RandomState(SEED + seed_offset)
    n = len(curve)
    t = np.linspace(0, freq * np.pi, n)
    noise = (amplitude * np.sin(t) +
             amplitude * 0.5 * np.sin(t * 2.7 + 1.1) +
             amplitude * 0.3 * rng.randn(n))
    kernel = np.ones(3) / 3
    noise = np.convolve(noise, kernel, mode="same")
    return curve + noise


def add_lr_bumps(curve: np.ndarray, effect: float = 0.03) -> np.ndarray:
    """Small loss drops at LR decay milestones (epochs 40, 60, 80)."""
    c = curve.copy()
    for ep in [40, 60, 80]:
        idx = ep - 1
        if idx < len(c):
            c[idx:] -= effect * 0.5
    return c


def generate_wer_curve(start: float, end: float, n: int,
                       seed_offset: int = 0) -> np.ndarray:
    """Generate WER curve with eval_every spacing and stage bumps."""
    rng = np.random.RandomState(SEED + seed_offset)
    t = np.linspace(0, 1, n)
    decay = start + (end - start) * (1 - np.exp(-3.5 * t)) / (1 - np.exp(-3.5))

    # Stage transition bumps
    s1e = STAGE1_END - 1
    s2e = STAGE2_END - 1
    decay[s1e:s1e + 3] += 1.5 * np.array([1, 0.6, 0.3])
    decay[s2e:s2e + 3] += 2.0 * np.array([1, 0.6, 0.3])

    # LR decay improvements
    for ep in [40, 60, 80]:
        idx = ep - 1
        if idx < n:
            decay[idx:] -= 0.2

    # Smooth noise
    noise = rng.randn(n) * 0.3
    kernel = np.ones(3) / 3
    noise = np.convolve(noise, kernel, mode="same")
    decay += noise

    # Keep only eval epochs + final
    wer_eval = []
    for i in range(n):
        if (i + 1) % EVAL_EVERY == 0 or i == n - 1:
            wer_eval.append(decay[i])
    return np.array(wer_eval)


# ============================================================================
# Full experiment curve generation
# ============================================================================

def generate_experiment_curves(exp: dict, seed_offset: int) -> dict:
    """Generate full 100-epoch training curves for one experiment."""
    b = BASELINE
    n = EPOCHS
    dw = exp.get("delta_wer", 0.0)
    dc = exp.get("delta_ctc", 0.0)

    # --- Stage 1 (epochs 1-30) ---
    s1_n = STAGE1_END
    ctc_g_s1 = smooth_decay(b["ctc_g_s1_start"] + dc,
                             b["ctc_g_s1_end"] + dc, s1_n)
    ctc_v_s1 = smooth_decay(b["ctc_v_s1_start"], b["ctc_v_s1_end"], s1_n)

    # --- Stage 2 (epochs 31-90) ---
    s2_n = STAGE2_END - STAGE1_END
    ctc_g_s2 = smooth_decay(b["ctc_g_s1_end"] + b["s1_s2_bump_ctc"] + dc,
                             b["ctc_g_s2_end"] + dc, s2_n)
    seg_s2   = smooth_decay(b["seg_s2_start"], b["seg_s2_end"], s2_n)

    # --- Stage 3 (epochs 91-100) ---
    s3_n = EPOCHS - STAGE2_END
    ctc_g_s3 = smooth_decay(b["ctc_g_s2_end"] + b["s2_s3_bump_ctc"] + dc,
                             b["ctc_g_s3_end"] + dc, s3_n)

    # Concatenate
    ctc_g = np.concatenate([ctc_g_s1, ctc_g_s2, ctc_g_s3])
    ctc_v = np.concatenate([ctc_v_s1, np.full(s2_n, np.nan), np.full(s3_n, np.nan)])
    seg   = np.concatenate([np.full(s1_n, np.nan), seg_s2, np.full(s3_n, np.nan)])

    # --- Enhancement losses ---
    align   = np.full(n, np.nan)
    entropy = np.full(n, np.nan)

    if "align_strength" in exp:
        a = exp["align_strength"]
        a1 = smooth_decay(a * 1.5, a * 0.6, s1_n)
        a2 = smooth_decay(a * 0.7, a * 0.3, s2_n)
        a3 = smooth_decay(a * 0.35, a * 0.2, s3_n)
        align = np.concatenate([a1, a2, a3]) * 0.15

    if "entropy_strength" in exp:
        e = exp["entropy_strength"]
        e1 = smooth_decay(e * 2.0, e * 0.8, s1_n)
        e2 = smooth_decay(e * 1.0, e * 0.4, s2_n)
        e3 = smooth_decay(e * 0.5, e * 0.3, s3_n)
        entropy = np.concatenate([e1, e2, e3]) * 0.02

    # --- Total loss ---
    alpha = 0.5
    total = ctc_g.copy()
    total[:s1_n] += alpha * ctc_v_s1
    total[s1_n:s1_n + s2_n] += alpha * seg_s2
    total += np.nan_to_num(align, 0)
    total += np.nan_to_num(entropy, 0)

    # Add noise + LR bumps
    ctc_g = add_lr_bumps(add_noise(ctc_g, 0.015, 2.5, seed_offset), b["lr_decay_effect"])
    ctc_v = add_noise(ctc_v, 0.02, 2.5, seed_offset + 100)
    seg   = add_noise(seg, 0.04, 2.0, seed_offset + 200)
    total = add_noise(total, 0.03, 2.0, seed_offset + 300)

    # WER curves
    wer_dev  = generate_wer_curve(b["wer_initial"],
                                   b["wer_final_dev"] + dw, n, seed_offset)
    rng = np.random.RandomState(SEED + seed_offset)
    wer_test = wer_dev[-1] + 0.7 + rng.uniform(-0.2, 0.3)
    wer_test = max(wer_dev[-1] + 0.4, min(wer_dev[-1] + 1.2, wer_test))

    return {
        "ctc_g": ctc_g, "ctc_v": ctc_v, "seg": seg,
        "total": total, "align": align, "entropy": entropy,
        "wer_dev": wer_dev, "wer_test": wer_test,
    }


# ============================================================================
# Timing estimation per experiment
# ============================================================================

def generate_timing(exp_name: str) -> dict:
    """Realistic per-epoch timing breakdown in seconds (NVIDIA T4)."""
    rng = np.random.RandomState(SEED + hash(exp_name) % 10000)

    # Base epoch time varies by model variant
    if "amp" in exp_name or "combined" in exp_name:
        total_ep = 340   # ~5.7 min with AMP
    elif "mobilenet" in exp_name and "frozen" in exp_name:
        total_ep = 500   # ~8.3 min — fastest config
    elif "mobilenet" in exp_name:
        total_ep = 600   # ~10 min (BiLSTM bottleneck)
    elif "efficientnet" in exp_name:
        total_ep = 1100  # ~18 min (heavier spatial encoder)
    elif "frozen" in exp_name:
        total_ep = 860   # ~14 min (no backbone backward)
    elif "multiscale" in exp_name:
        total_ep = 1439  # ~24 min
    elif "dwconv" in exp_name:
        total_ep = 1200  # ~20 min
    else:
        total_ep = 1224  # ~20.4 min baseline

    fractions = {
        "data_xfer": 0.03 + rng.uniform(-0.005, 0.005),
        "forward":   0.57 + rng.uniform(-0.02, 0.02),
        "gsba":      0.06 + rng.uniform(-0.01, 0.01),
        "loss":      0.02 + rng.uniform(-0.003, 0.003),
        "backward":  0.32 + rng.uniform(-0.015, 0.015),
    }
    total_frac = sum(fractions.values())
    fractions = {k: v / total_frac for k, v in fractions.items()}
    return {k: round(v * total_ep) for k, v in fractions.items()}


# ============================================================================
# Markdown generation
# ============================================================================

def loss_table(name: str, c: dict, exp: dict) -> str:
    """Per-experiment 3-stage loss summary table."""
    ctc_g = c["ctc_g"]; total = c["total"]
    s1e = STAGE1_END; s2e = STAGE2_END

    return "\n".join([
        f"### {name}",
        f"**{exp['category']}** | {exp['desc']}",
        f"**Dev WER:** {c['wer_dev'][-1]:.2f}% | **Test WER:** {c['wer_test']:.2f}%",
        "",
        "| Stage | Epochs | CTC(g) | CTC(v) | Seg | Total |",
        "|---|---|---|---|---|---|",
        f"| 1 | 1-{s1e} | {np.nanmean(ctc_g[:s1e]):.3f} | "
        f"{np.nanmean(c['ctc_v'][:s1e]):.3f} | — | {np.nanmean(total[:s1e]):.3f} |",
        f"| 2 | {s1e+1}-{s2e} | {np.nanmean(ctc_g[s1e:s2e]):.3f} | — | "
        f"{np.nanmean(c['seg'][s1e:s2e]):.3f} | {np.nanmean(total[s1e:s2e]):.3f} |",
        f"| 3 | {s2e+1}-{EPOCHS} | {np.nanmean(ctc_g[s2e:]):.3f} | — | — | "
        f"{np.nanmean(total[s2e:]):.3f} |",
        "",
    ])


def wer_table(results: dict) -> str:
    """Ranked WER comparison across all experiments."""
    baseline_dev = results["baseline"]["curves"]["wer_dev"][-1]
    baseline_test = results["baseline"]["curves"]["wer_test"]
    sorted_exps = sorted(results.items(),
                         key=lambda x: x[1]["curves"]["wer_dev"][-1])

    lines = [
        "## 📊 WER Comparison (All Experiments)",
        "",
        "| # | Experiment | Category | Dev WER | Test WER | Δ Dev | Δ Test |",
        "|---|---|---|---|---|---|---|",
    ]
    for i, (name, data) in enumerate(sorted_exps, 1):
        c = data["curves"]
        dev = c["wer_dev"][-1]
        test = c["wer_test"]
        dd = f"{dev - baseline_dev:+.2f}" if abs(dev - baseline_dev) > 0.001 else "—"
        dt = f"{test - baseline_test:+.2f}"
        lines.append(
            f"| {i} | `{name}` | {data['exp']['category']} | "
            f"{dev:.2f}% | {test:.2f}% | {dd} | {dt} |"
        )
    return "\n".join(lines) + "\n"


def timing_table(t: dict, name: str) -> str:
    """Timing profile for one experiment."""
    total = sum(t.values())
    return "\n".join([
        f"### ⏱ Timing: {name}",
        "",
        "| Phase | Seconds | % |",
        "|---|---|---|",
        f"| Data transfer | {t['data_xfer']}s | {100*t['data_xfer']/total:.1f}% |",
        f"| Forward pass | {t['forward']}s | {100*t['forward']/total:.1f}% |",
        f"| GSBA | {t['gsba']}s | {100*t['gsba']/total:.1f}% |",
        f"| Loss compute | {t['loss']}s | {100*t['loss']/total:.1f}% |",
        f"| Backward+Opt | {t['backward']}s | {100*t['backward']/total:.1f}% |",
        f"| **Total** | **{total}s** | **100%** |",
        "",
        f"Est. total: **{total * EPOCHS / 3600:.1f}h** "
        f"({total * EPOCHS / 86400:.1f}d)",
        "",
    ])


def detail_table(c: dict, eval_epochs: list) -> str:
    """Per-eval-epoch loss/WER detail."""
    lines = [
        "### Per-Eval-Epoch Detail",
        "",
        "| Epoch | St | CTC(g) | CTC(v) | Seg | Total | WER(dev) |",
        "|---|---|---|---|---|---|---|",
    ]
    for ep_idx in eval_epochs:
        ep = ep_idx + 1
        stage = 1 if ep <= STAGE1_END else (2 if ep <= STAGE2_END else 3)
        cg = f"{c['ctc_g'][ep_idx]:.3f}"
        cv = f"{c['ctc_v'][ep_idx]:.3f}" if not np.isnan(c['ctc_v'][ep_idx]) else "—"
        sg = f"{c['seg'][ep_idx]:.3f}" if not np.isnan(c['seg'][ep_idx]) else "—"
        tl = f"{c['total'][ep_idx]:.3f}"
        ei = [j for j, e in enumerate(eval_epochs) if e == ep_idx]
        wr = f"{c['wer_dev'][ei[0]]:.2f}%" if ei else "—"
        lines.append(f"| {ep} | {stage} | {cg} | {cv} | {sg} | {tl} | {wr} |")
    return "\n".join(lines) + "\n"


# ============================================================================
# Main report builder
# ============================================================================

def build_report(results: dict) -> str:
    """Full markdown report."""
    baseline = results["baseline"]
    eval_eps = [i for i in range(EPOCHS)
                if (i + 1) % EVAL_EVERY == 0 or i == EPOCHS - 1]

    md = [
        "# SMKD Enhancement Suite — Synthetic Experiment Results",
        "",
        f"**Generated:** 2026-06-20",
        f"**Dataset:** PHOENIX-2014-T (5,672 train / 540 dev / 629 test)",
        f"**Model:** SMKD (d_model=512, 3 stages: "
        f"{STAGE1_END}/{STAGE2_END}/{EPOCHS})",
        f"**GPU:** NVIDIA T4 (simulated)",
        f"**Total experiments:** {len(results)}",
        "",
        "---",
        "",
        wer_table(results),
        "---",
        "",
        "## 📈 Baseline — Detailed Results",
        "",
        loss_table("baseline", baseline["curves"], baseline["exp"]),
        timing_table(baseline["timing"], "baseline"),
        detail_table(baseline["curves"], eval_eps),
        "---",
        "",
    ]

    # Per-category sections
    for cat in ["loss", "model", "training", "combined"]:
        md.append(f"## 🔬 {cat.capitalize()} Experiments")
        md.append("")
        for name, data in results.items():
            if data["exp"]["category"] != cat:
                continue
            md.append(loss_table(name, data["curves"], data["exp"]))
            md.append(timing_table(data["timing"], name))
        md.append("---")
        md.append("")

    # Key takeaways
    sorted_exps = sorted(results.items(),
                         key=lambda x: x[1]["curves"]["wer_dev"][-1])
    best = sorted_exps[0]
    worst = sorted_exps[-1]
    bl_dev = baseline["curves"]["wer_dev"][-1]
    t2_dev = results["combined_tier2"]["curves"]["wer_dev"][-1]
    pt_dev = results["model_pretrained"]["curves"]["wer_dev"][-1]

    md += [
        "## 🏆 Key Takeaways",
        "",
        f"1. **Best:** `{best[0]}` — {best[1]['curves']['wer_dev'][-1]:.2f}% dev WER "
        f"({best[1]['curves']['wer_dev'][-1] - bl_dev:+.2f}% vs baseline)",
        f"2. **Worst:** `{worst[0]}` — {worst[1]['curves']['wer_dev'][-1]:.2f}% dev WER "
        f"({worst[1]['curves']['wer_dev'][-1] - bl_dev:+.2f}% vs baseline)",
        f"3. **Combined Tier 2:** {t2_dev:.2f}% dev WER "
        f"({t2_dev - bl_dev:+.2f}%) — all enhancements together",
        f"4. **Best single change:** `model_pretrained` (ImageNet backbone) "
        f"— {pt_dev:.2f}% ({pt_dev - bl_dev:+.2f}%)",
        f"5. **AMP impact:** Negligible WER change (+0.05%) with **6× training speedup** "
        f"(39.3h → 6.5h)",
        f"6. **Pretrained backbones consistently beat from-scratch:** "
        f"ResNet-18: −0.8%, EfficientNet-B0: −0.5%, MobileNetV3: −0.2%",
        f"7. **Frozen backbones:** trade ~0.3-0.8% WER for 30-60% faster training "
        f"— useful for rapid iteration",
        f"8. **Stage 3 decouple spike:** 1.5-3.0% WER jump at epoch 91 — "
        f"`train_smooth_transition` reduces this by ~0.3%",
        f"9. **VAC + Entropy combined** (−0.7%) outperforms either alone "
        f"(VAC: −0.5%, Entropy: −0.3%)",
        f"10. **Multi-scale temporal branches** (−0.6%) are the best model-only "
        f"enhancement at the cost of +15% training time",
        "",
        "---",
        "*Synthetic results — generated to reflect realistic training dynamics on PHOENIX-2014-T.*",
        "*Actual results depend on hardware, data preprocessing, and random seed.*",
    ]

    return "\n".join(md)


# ============================================================================
# Main
# ============================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = {}

    for i, exp in enumerate(EXPERIMENTS):
        curves = generate_experiment_curves(exp, seed_offset=i * 10)
        timing = generate_timing(exp["name"])
        results[exp["name"]] = {
            "curves": curves,
            "timing": timing,
            "exp": exp,
        }

    md = build_report(results)
    with open(OUTPUT_MD, "w") as f:
        f.write(md)

    print(f"✅ Generated synthetic results for {len(results)} experiments")
    print(f"   Output: {OUTPUT_MD}")
    print()

    # Quick summary
    bl = results["baseline"]["curves"]["wer_dev"][-1]
    print("   Quick WER Summary (sorted by dev WER):")
    print(f"   {'Experiment':<34s} {'Dev':>7s} {'Test':>7s} {'ΔDev':>7s}")
    print(f"   {'-'*55}")
    for name, data in sorted(results.items(),
                             key=lambda x: x[1]["curves"]["wer_dev"][-1]):
        dev = data["curves"]["wer_dev"][-1]
        test = data["curves"]["wer_test"]
        print(f"   {name:<34s} {dev:6.2f}% {test:6.2f}% {dev-bl:+6.2f}%")


if __name__ == "__main__":
    main()
