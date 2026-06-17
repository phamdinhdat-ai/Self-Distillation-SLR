#!/usr/bin/env python3
"""
SMKD Experiment Benchmark Runner
=================================
Runs ALL individual experiment configs (except combined tiers) with synthetic
data through full 3-stage training, then produces a unified comparison report:
  - Console comparison table (ranked by best WER)
  - Summary CSV (runs/benchmark/summary.csv)
  - Comparison plots (runs/benchmark/comparison_plots.png)

Usage:
    # Full benchmark (all 17 configs, 100 epochs each)
    python experiments/benchmark.py

    # Quick benchmark (subset, 20 epochs)
    python experiments/benchmark.py --epochs 20

    # Specific configs only
    python experiments/benchmark.py --configs baseline loss_vac loss_entropy

    # Custom output directory
    python experiments/benchmark.py --output-dir runs/my_benchmark
"""

import argparse
import csv
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trainer import Trainer


# ---------------------------------------------------------------------------
# Config loading & flattening (self-contained; mirrors experiments/runner.py)
# ---------------------------------------------------------------------------

def load_yaml_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def flatten_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flatten a nested YAML config into the flat dict expected by Trainer(cfg).
    """
    flat: Dict[str, Any] = {}

    # Data
    data = cfg.get("data", {})
    flat["data_root"] = data.get("data_root", "./data/phoenix")
    flat["img_size"] = data.get("img_size", 224)
    flat["max_frames"] = data.get("max_frames", 300)
    flat["num_workers"] = data.get("num_workers", 0)
    flat["batch_size"] = data.get("batch_size", 2)
    flat["synthetic"] = data.get("synthetic", True)

    ta = data.get("temporal_aug", {})
    flat["temporal_aug_enabled"] = ta.get("enabled", False)
    flat["temporal_mask_prob"] = ta.get("temporal_mask_prob", 0.3)
    flat["temporal_mask_max_ratio"] = ta.get("temporal_mask_max_ratio", 0.15)
    flat["temporal_jitter_range"] = ta.get("temporal_jitter_range", 0.0)

    # Model
    model = cfg.get("model", {})
    flat["d_model"] = model.get("d_model", 512)
    flat["hidden_size"] = model.get("hidden_size", 512)
    flat["backbone"] = model.get("backbone", "resnet18")
    flat["pretrained_backbone"] = model.get("pretrained_backbone", False)
    flat["freeze_backbone"] = model.get("freeze_backbone", False)
    flat["temporal_conv_type"] = model.get("temporal_conv_type", "standard")
    flat["multi_scale_temporal"] = model.get("multi_scale_temporal", False)
    flat["multi_scale_dilation_rates"] = model.get("multi_scale_dilation_rates", [1, 2])
    flat["use_tsm"] = model.get("use_tsm", False)

    # Loss
    loss = cfg.get("loss", {})
    flat["alpha"] = loss.get("alpha", 0.5)
    flat["label_smoothing"] = loss.get("label_smoothing", 0.2)
    flat["alpha_schedule"] = loss.get("alpha_schedule", "fixed")
    flat["alpha_start"] = loss.get("alpha_start", 0.1)
    flat["alpha_end"] = loss.get("alpha_end", 0.5)

    fa = loss.get("feature_alignment", {})
    flat["feature_alignment_enabled"] = fa.get("enabled", False)
    flat["feature_alignment_weight"] = fa.get("weight", 0.1)
    flat["feature_alignment_mode"] = fa.get("mode", "cosine")

    er = loss.get("entropy_regularization", {})
    flat["entropy_reg_enabled"] = er.get("enabled", False)
    flat["entropy_reg_weight"] = er.get("weight", 0.01)

    fc = loss.get("focal_ctc", {})
    flat["focal_ctc_enabled"] = fc.get("enabled", False)
    flat["focal_ctc_gamma"] = fc.get("gamma", 2.0)

    flat["gsba_confidence_weight"] = loss.get("gsba_confidence_weight", False)

    # Schedule
    sched = cfg.get("schedule", {})
    flat["epochs"] = sched.get("epochs", 100)
    flat["stage1_end"] = sched.get("stage1_end", 30)
    flat["stage2_end"] = sched.get("stage2_end", 90)
    flat["eval_every"] = sched.get("eval_every", 5)
    flat["gsba_update_every"] = sched.get("gsba_update_every", 10)
    flat["smooth_transitions"] = sched.get("smooth_transitions", False)
    flat["smooth_transition_epochs"] = sched.get("smooth_transition_epochs", 5)

    sd = sched.get("self_distill", {})
    flat["self_distill_enabled"] = sd.get("enabled", False)
    flat["self_distill_teacher_ckpt"] = sd.get("teacher_ckpt", "")
    flat["kd_temperature"] = sd.get("kd_temperature", 4.0)
    flat["kd_weight"] = sd.get("kd_weight", 0.5)

    # Optimization
    opt = cfg.get("optimization", {})
    flat["lr"] = opt.get("lr", 1e-4)
    flat["lr_milestones"] = opt.get("lr_milestones", [40, 60, 80])
    flat["lr_gamma"] = opt.get("lr_gamma", 0.5)
    flat["use_amp"] = opt.get("use_amp", False)
    flat["grad_accum_steps"] = opt.get("grad_accum_steps", 1)
    flat["use_ema"] = opt.get("use_ema", False)
    flat["ema_decay"] = opt.get("ema_decay", 0.999)

    # Hardware
    hw = cfg.get("hardware", {})
    flat["multi_gpu"] = hw.get("multi_gpu", True)

    # Output
    out = cfg.get("output", {})
    flat["ckpt_dir"] = out.get("ckpt_dir", "./checkpoints")
    flat["experiment_name"] = out.get("experiment_name", "default")

    return flat


def save_history_csv(history: Dict, path: str):
    """Save per-epoch history dict to CSV."""
    epochs = history["epoch"]
    if not epochs:
        return
    keys = [k for k in history.keys() if k != "epoch"]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch"] + keys)
        for i, ep in enumerate(epochs):
            row = [ep] + [history[k][i] if history[k][i] is not None else "" for k in keys]
            writer.writerow(row)


def setup_run_dir(experiment_name: str, timestamp: str = None) -> str:
    """Create runs/<experiment_name>/<timestamp>/ and return its path."""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("runs", experiment_name, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

# ============================================================================
# Constants
# ============================================================================

CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "configs", "experiments",
)

# Configs to exclude (combinations, not individual ablations)
EXCLUDE_CONFIGS = {"combined_tier1.yaml", "combined_tier2.yaml"}

# Human-readable descriptions for each experiment
EXPERIMENT_DESCRIPTIONS = {
    "baseline":                "Baseline (paper reproduction)",
    "loss_vac":                "VAC-style visual alignment constraint",
    "loss_entropy":            "Entropy regularization (anti-spike)",
    "loss_vac_entropy":        "VAC + entropy combined",
    "loss_confidence_gsba":    "Confidence-weighted GSBA",
    "loss_curriculum_alpha":   "Curriculum schedule on α",
    "loss_focal_ctc":          "Focal CTC modulation",
    "model_dwconv":            "Depthwise-separable temporal convs",
    "model_multiscale":        "Multi-scale temporal branches",
    "model_mobilenet":         "MobileNetV3-Small backbone",
    "model_tsm":               "Temporal Shift Module (TSM)",
    "model_pretrained":        "ImageNet pretrained backbone",
    "model_frozen_backbone":   "Frozen pretrained backbone",
    "train_amp":               "AMP + gradient accumulation",
    "train_ema":               "Exponential moving average (EMA)",
    "train_temporal_aug":      "Temporal data augmentation",
    "train_smooth_transition": "Smooth stage transitions",
}

# ============================================================================
# Config discovery
# ============================================================================

def discover_configs(
    config_dir: str = CONFIG_DIR,
    include_filter: Optional[List[str]] = None,
) -> List[Tuple[str, str]]:
    """
    Discover experiment config files.

    Returns:
        List of (experiment_name, config_path) tuples, sorted.
    """
    configs = []
    for fname in sorted(os.listdir(config_dir)):
        if not fname.endswith(".yaml"):
            continue
        if fname in EXCLUDE_CONFIGS:
            continue

        name = fname.replace(".yaml", "")
        if include_filter and name not in include_filter:
            continue

        configs.append((name, os.path.join(config_dir, fname)))

    return configs


# ============================================================================
# Enhancement extraction
# ============================================================================

def extract_enhancements(flat_cfg: Dict[str, Any]) -> List[str]:
    """
    Extract a human-readable list of enabled enhancements from a flat config.
    """
    enhancements = []

    # Loss enhancements
    if flat_cfg.get("feature_alignment_enabled", False):
        mode = flat_cfg.get("feature_alignment_mode", "cosine")
        w = flat_cfg.get("feature_alignment_weight", 0.1)
        enhancements.append(f"VAC({mode},w={w})")

    if flat_cfg.get("entropy_reg_enabled", False):
        w = flat_cfg.get("entropy_reg_weight", 0.01)
        enhancements.append(f"EntropyReg(w={w})")

    if flat_cfg.get("focal_ctc_enabled", False):
        gamma = flat_cfg.get("focal_ctc_gamma", 2.0)
        enhancements.append(f"FocalCTC(γ={gamma})")

    if flat_cfg.get("gsba_confidence_weight", False):
        enhancements.append("ConfWeightGSBA")

    alpha_sched = flat_cfg.get("alpha_schedule", "fixed")
    if alpha_sched != "fixed":
        a_start = flat_cfg.get("alpha_start", 0.1)
        a_end = flat_cfg.get("alpha_end", 0.5)
        enhancements.append(f"CurrAlpha({alpha_sched},{a_start}→{a_end})")

    # Model enhancements
    if flat_cfg.get("temporal_conv_type", "standard") == "depthwise_separable":
        enhancements.append("DWConv")

    if flat_cfg.get("multi_scale_temporal", False):
        dr = flat_cfg.get("multi_scale_dilation_rates", [1, 2])
        enhancements.append(f"MultiScale(dr={dr})")

    backbone = flat_cfg.get("backbone", "resnet18")
    if backbone != "resnet18":
        enhancements.append(f"Backbone:{backbone}")

    if flat_cfg.get("use_tsm", False):
        enhancements.append("TSM")

    if flat_cfg.get("pretrained_backbone", False):
        enhancements.append("ImageNetPretrained")

    if flat_cfg.get("freeze_backbone", False):
        enhancements.append("FrozenBackbone")

    # Training enhancements
    if flat_cfg.get("use_amp", False):
        gas = flat_cfg.get("grad_accum_steps", 1)
        enhancements.append(f"AMP(gas={gas})")

    if flat_cfg.get("use_ema", False):
        decay = flat_cfg.get("ema_decay", 0.999)
        enhancements.append(f"EMA(decay={decay})")

    if flat_cfg.get("temporal_aug_enabled", False):
        mp = flat_cfg.get("temporal_mask_prob", 0.3)
        mr = flat_cfg.get("temporal_mask_max_ratio", 0.15)
        jr = flat_cfg.get("temporal_jitter_range", 0.0)
        enhancements.append(f"TempAug(mask={mp},mr={mr},jitter={jr})")

    if flat_cfg.get("smooth_transitions", False):
        ste = flat_cfg.get("smooth_transition_epochs", 5)
        enhancements.append(f"SmoothTrans(ep={ste})")

    return enhancements if enhancements else ["(none)"]


# ============================================================================
# Per-experiment runner
# ============================================================================

def run_single_experiment(
    name: str,
    config_path: str,
    output_root: str,
    override_epochs: Optional[int] = None,
    data_root: str = "./data/phoenix",
) -> Dict[str, Any]:
    """
    Run a single experiment config through full training.

    Returns:
        Dict with all metrics for the summary table, or {"status": "FAILED", "error": str}
    """
    result: Dict[str, Any] = {"experiment_name": name, "status": "OK"}

    # Load and flatten config
    try:
        cfg = load_yaml_config(config_path)
        flat = flatten_config(cfg)
    except Exception as e:
        result["status"] = "FAILED"
        result["error"] = f"Config load error: {e}"
        return result

    # Enforce synthetic data
    flat["synthetic"] = True
    flat["data_root"] = data_root
    flat["multi_gpu"] = False  # avoid DataParallel complications in benchmark

    # Override epochs if requested
    if override_epochs is not None:
        flat["epochs"] = override_epochs
        # Proportionally scale stage boundaries
        orig_epochs = cfg.get("schedule", {}).get("epochs", 100)
        ratio = override_epochs / max(orig_epochs, 1)
        flat["stage1_end"] = max(1, int(cfg.get("schedule", {}).get("stage1_end", 30) * ratio))
        flat["stage2_end"] = max(flat["stage1_end"] + 1,
                                 int(cfg.get("schedule", {}).get("stage2_end", 90) * ratio))
        flat["eval_every"] = max(1, int(cfg.get("schedule", {}).get("eval_every", 5) * ratio))
        flat["gsba_update_every"] = max(1, int(cfg.get("schedule", {}).get("gsba_update_every", 10) * ratio))
        # Scale LR milestones
        flat["lr_milestones"] = [
            max(1, int(m * ratio))
            for m in cfg.get("optimization", {}).get("lr_milestones", [40, 60, 80])
        ]

    # Determine experiment name for run directory
    exp_name = flat.get("experiment_name", name)

    # Extract metadata before training
    result["description"] = EXPERIMENT_DESCRIPTIONS.get(name, name)
    result["enhancements"] = extract_enhancements(flat)
    result["backbone"] = flat.get("backbone", "resnet18")
    result["d_model"] = flat.get("d_model", 512)
    result["hidden_size"] = flat.get("hidden_size", 512)
    result["epochs"] = flat["epochs"]

    # Setup run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_root, exp_name, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    result["run_dir"] = run_dir

    # Save config copy
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    with open(os.path.join(run_dir, "flat_config.yaml"), "w") as f:
        yaml.dump(flat, f, default_flow_style=False)

    # ---- Run training ----
    trainer = None
    train_start = time.time()
    try:
        trainer = Trainer(flat)
        result["total_params"] = sum(p.numel() for p in trainer.raw_model.parameters())
        result["trainable_params"] = sum(
            p.numel() for p in trainer.raw_model.parameters() if p.requires_grad
        )
        result["frozen_params"] = result["total_params"] - result["trainable_params"]

        trainer.train()
    except Exception as e:
        result["status"] = "FAILED"
        result["error"] = f"Training error: {e}\n{traceback.format_exc()}"
        result["total_time"] = time.time() - train_start
        return result

    total_time = time.time() - train_start
    result["total_time"] = total_time

    # ---- Collect metrics from trainer ----
    result["best_wer"] = trainer.best_wer if trainer.best_wer < float("inf") else None
    result["best_wer_epoch"] = trainer.best_epoch

    # Final epoch metrics
    history = trainer.history
    n_epochs = len(history["epoch"])
    if n_epochs > 0:
        wer_vals = [w for w in history["wer"] if w is not None]
        result["final_wer"] = wer_vals[-1] if wer_vals else None

        for key in ("total", "ctc_g", "ctc_v", "seg", "align", "entropy", "kd"):
            vals = [v for v in history[key] if v is not None]
            result[f"final_loss_{key}"] = vals[-1] if vals else None

        time_vals = [t for t in history["time"] if t is not None]
        result["avg_time_per_epoch"] = sum(time_vals) / len(time_vals) if time_vals else 0.0
    else:
        result["final_wer"] = None
        result["avg_time_per_epoch"] = 0.0

    # Save per-run history CSV
    history_path = os.path.join(run_dir, "history.csv")
    save_history_csv(history, history_path)

    # Save per-run WER plot
    try:
        plot_path = os.path.join(run_dir, "wer_history.png")
        trainer.plot_history(save_path=plot_path, show=False)
    except Exception:
        pass  # plot failure is non-fatal

    return result


# ============================================================================
# Formatting helpers
# ============================================================================

def fmt_seconds(seconds: float) -> str:
    """Format seconds as human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s"


def fmt_wer(wer: Optional[float]) -> str:
    if wer is None or wer == float("inf"):
        return "   --   "
    return f"{wer:7.2f}%"


def fmt_params(n: Optional[int]) -> str:
    if n is None:
        return "     --"
    if n >= 1_000_000:
        return f"{n/1_000_000:5.1f}M"
    return f"{n/1_000:5.0f}K"


# ============================================================================
# Summary table
# ============================================================================

def print_summary_table(results: List[Dict[str, Any]]):
    """Print a formatted comparison table, ranked by best WER."""
    if not results:
        print("\n  No results to display.")
        return

    # Sort: OK first by best WER ascending, then FAILED at bottom
    def sort_key(r):
        if r.get("status") != "OK":
            return (1, 0)
        wer = r.get("best_wer")
        return (0, wer if wer is not None else float("inf"))

    sorted_results = sorted(results, key=sort_key)

    print()
    print("=" * 120)
    header = (f"  {'Experiment':<28s} {'Status':>6s} {'Backbone':<16s} "
              f"{'Params':>8s} {'Best WER':>9s} {'@Ep':>5s} "
              f"{'Final WER':>9s} {'Loss':>8s} {'Time':>10s}")
    print(header)
    print("-" * 120)

    for r in sorted_results:
        name = r["experiment_name"][:27]
        status = r.get("status", "??")
        backbone = r.get("backbone", "resnet18")[:15]
        params = fmt_params(r.get("total_params"))
        best_wer = fmt_wer(r.get("best_wer"))
        best_ep = f"{r.get('best_wer_epoch', '--'):>5}" if r.get("best_wer_epoch") else "   --"
        final_wer = fmt_wer(r.get("final_wer"))
        loss = f"{r.get('final_loss_total', 0):8.4f}" if r.get("final_loss_total") is not None else "     --"
        ttime = fmt_seconds(r.get("total_time", 0))

        line = (f"  {name:<28s} {status:>6s} {backbone:<16s} "
                f"{params:>8s} {best_wer:>9s} {best_ep:>5s} "
                f"{final_wer:>9s} {loss:>8s} {ttime:>10s}")
        print(line)

    print("-" * 120)

    # Print enhancement details
    print()
    print("  Enhancement details:")
    for r in sorted_results:
        enh = ", ".join(r.get("enhancements", []))
        print(f"    {r['experiment_name']:<28s} → {enh}")
    print("=" * 120)
    print()


# ============================================================================
# Comparison plots
# ============================================================================

def generate_comparison_plots(results: List[Dict[str, Any]], output_path: str):
    """
    Generate a 2×2 comparison figure:
      - Top-left:  WER overlay (all successful runs)
      - Top-right:  Total loss overlay
      - Bottom-left:  Best WER bar chart (sorted)
      - Bottom-right: Training time bar chart (sorted)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import csv as csv_module

    ok_results = [r for r in results if r.get("status") == "OK"]
    if not ok_results:
        print("  No successful runs to plot.")
        return

    n = len(ok_results)
    colors = plt.cm.tab20(range(n))

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # ---- Top-left: WER overlay ----
    ax = axes[0, 0]
    for i, r in enumerate(ok_results):
        hist_path = os.path.join(r["run_dir"], "history.csv")
        wer_epochs, wer_vals = _load_history_column(hist_path, "wer")
        if wer_vals:
            ax.plot(wer_epochs, wer_vals, marker=".", color=colors[i],
                    label=r["experiment_name"], linewidth=1.5, markersize=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("WER (%)")
    ax.set_title("WER over epochs (synthetic data)")
    if n <= 20:
        ax.legend(fontsize=6, loc="upper right", ncol=2)
    ax.grid(alpha=0.3)

    # ---- Top-right: Total loss overlay ----
    ax = axes[0, 1]
    for i, r in enumerate(ok_results):
        hist_path = os.path.join(r["run_dir"], "history.csv")
        loss_epochs, loss_vals = _load_history_column(hist_path, "total")
        if loss_vals:
            ax.plot(loss_epochs, loss_vals, color=colors[i],
                    label=r["experiment_name"], linewidth=1.5, alpha=0.85)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Total Loss")
    ax.set_title("Training loss over epochs")
    if n <= 20:
        ax.legend(fontsize=6, loc="upper right", ncol=2)
    ax.grid(alpha=0.3)

    # ---- Bottom-left: Best WER bar chart ----
    ax = axes[1, 0]
    sorted_by_wer = sorted(ok_results, key=lambda r: r.get("best_wer") or float("inf"))
    names = [r["experiment_name"] for r in sorted_by_wer]
    wers = [r.get("best_wer") or 0 for r in sorted_by_wer]
    bar_colors = [colors[ok_results.index(r)] for r in sorted_by_wer]
    bars = ax.barh(range(len(names)), wers, color=bar_colors, edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("Best WER (%)")
    ax.set_title("Best WER by experiment (lower is better)")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    # Annotate bars
    for bar, wer in zip(bars, wers):
        if wer > 0:
            ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                    f"{wer:.1f}%", va="center", fontsize=7)

    # ---- Bottom-right: Training time bar chart ----
    ax = axes[1, 1]
    sorted_by_time = sorted(ok_results, key=lambda r: r.get("total_time", 0))
    names_t = [r["experiment_name"] for r in sorted_by_time]
    times = [r.get("total_time", 0) for r in sorted_by_time]
    t_colors = [colors[ok_results.index(r)] for r in sorted_by_time]
    bars_t = ax.barh(range(len(names_t)), times, color=t_colors, edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(len(names_t)))
    ax.set_yticklabels(names_t, fontsize=7)
    ax.set_xlabel("Training time (seconds)")
    ax.set_title("Total training time by experiment")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    for bar, t in zip(bars_t, times):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                fmt_seconds(t), va="center", fontsize=7)

    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Comparison plots saved to: {output_path}")


def _load_history_column(hist_path: str, column: str) -> Tuple[List[int], List[float]]:
    """Load a column from a history CSV, returning (epochs, values) where both are non-None."""
    epochs, vals = [], []
    if not os.path.isfile(hist_path):
        return epochs, vals
    try:
        with open(hist_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ep_str = row.get("epoch", "").strip()
                val_str = row.get(column, "").strip()
                if ep_str and val_str:
                    try:
                        epochs.append(int(float(ep_str)))
                        vals.append(float(val_str))
                    except (ValueError, TypeError):
                        pass
    except Exception:
        pass
    return epochs, vals


# ============================================================================
# Summary CSV
# ============================================================================

def save_summary_csv(results: List[Dict[str, Any]], output_path: str):
    """Save a unified summary CSV with one row per experiment."""
    if not results:
        return

    # Determine all columns from the first result (plus standard ones)
    standard_cols = [
        "experiment_name", "description", "status", "error",
        "backbone", "d_model", "hidden_size", "epochs",
        "total_params", "trainable_params", "frozen_params",
        "best_wer", "best_wer_epoch", "final_wer",
        "final_loss_total", "final_loss_ctc_g", "final_loss_ctc_v",
        "final_loss_seg", "final_loss_align", "final_loss_entropy",
        "total_time", "avg_time_per_epoch",
        "enhancements", "run_dir",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=standard_cols, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            # Flatten enhancements list to string
            row = dict(r)
            if isinstance(row.get("enhancements"), list):
                row["enhancements"] = "; ".join(row["enhancements"])
            writer.writerow(row)

    print(f"  Summary CSV saved to: {output_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="SMKD Experiment Benchmark — batch-run all configs with synthetic data"
    )
    parser.add_argument(
        "--configs", nargs="*", default=None,
        help="Specific experiment names to run (default: all except combined tiers)",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override epoch count for all runs (default: use config's value = 100)",
    )
    parser.add_argument(
        "--output-dir", default="runs/benchmark",
        help="Root output directory (default: runs/benchmark)",
    )
    parser.add_argument(
        "--data-root", default="./data/phoenix",
        help="PHOENIX-2014T data root (default: ./data/phoenix)",
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip generating comparison plots",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip experiments that already have a run directory under output-dir",
    )
    args = parser.parse_args()

    # ---- Discover configs ----
    configs = discover_configs(CONFIG_DIR, args.configs)
    if not configs:
        print("No experiment configs found. Aborting.")
        return

    n_total = len(configs)
    print()
    print("=" * 72)
    print(f"  SMKD Experiment Benchmark — {n_total} configs")
    print(f"  Output root: {args.output_dir}")
    print(f"  Epochs override: {args.epochs if args.epochs else 'use config default (100)'}")
    print(f"  Data: synthetic only")
    print("=" * 72)

    # Print config list
    print()
    print("  Experiments to run:")
    for i, (name, path) in enumerate(configs):
        desc = EXPERIMENT_DESCRIPTIONS.get(name, name)
        print(f"    [{i+1:2d}/{n_total}] {name:<28s} — {desc}")
    print()

    # ---- Run all experiments ----
    results: List[Dict[str, Any]] = []
    benchmark_start = time.time()
    success_count = 0
    fail_count = 0

    for i, (name, config_path) in enumerate(configs):
        progress = f"[{i+1}/{n_total}]"
        desc = EXPERIMENT_DESCRIPTIONS.get(name, name)

        print("-" * 72)
        print(f"  {progress} {name} — {desc}")
        print(f"  Config: {config_path}")
        print(f"  Start:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("-" * 72)

        run_start = time.time()

        # Check skip-existing (look for history.csv in any timestamped subdir)
        if args.skip_existing:
            exp_output_dir = os.path.join(args.output_dir, name)
            if os.path.isdir(exp_output_dir):
                # Check if any timestamped subdirectory contains history.csv
                has_existing = False
                for sub in os.listdir(exp_output_dir):
                    sub_path = os.path.join(exp_output_dir, sub)
                    if os.path.isdir(sub_path) and os.path.isfile(
                        os.path.join(sub_path, "history.csv")
                    ):
                        has_existing = True
                        break
                if has_existing:
                    print(f"  ⏭  Skipping (existing run found under {exp_output_dir})")
                    continue

        result = run_single_experiment(
            name=name,
            config_path=config_path,
            output_root=args.output_dir,
            override_epochs=args.epochs,
            data_root=args.data_root,
        )
        results.append(result)

        run_time = time.time() - run_start

        if result.get("status") == "OK":
            success_count += 1
            best_wer = result.get("best_wer")
            best_wer_str = f"{best_wer:.2f}%" if best_wer and best_wer < float("inf") else "--"
            print(f"  ✓  Completed in {fmt_seconds(run_time)}  |  "
                  f"Best WER: {best_wer_str}  |  "
                  f"Run dir: {result.get('run_dir', 'N/A')}")
        else:
            fail_count += 1
            error_msg = result.get("error", "Unknown error")
            # Print only first 2 lines of error
            error_short = "\n".join(error_msg.split("\n")[:2])
            print(f"  ✗  FAILED after {fmt_seconds(run_time)}")
            print(f"     Error: {error_short}")

        # Progress estimate
        elapsed_total = time.time() - benchmark_start
        avg_per_run = elapsed_total / (i + 1)
        remaining = avg_per_run * (n_total - i - 1)
        print(f"  ⏱  Elapsed: {fmt_seconds(elapsed_total)}  |  "
              f"Est. remaining: {fmt_seconds(remaining)}  |  "
              f"Success: {success_count}/{i+1}")

    # ---- Final report ----
    total_benchmark_time = time.time() - benchmark_start
    print()
    print("=" * 72)
    print(f"  BENCHMARK COMPLETE")
    print(f"  Total time: {fmt_seconds(total_benchmark_time)}")
    print(f"  Successful: {success_count}/{n_total}")
    print(f"  Failed:     {fail_count}/{n_total}")
    print("=" * 72)

    # Print summary table
    print_summary_table(results)

    # Save summary CSV
    summary_csv = os.path.join(args.output_dir, "summary.csv")
    save_summary_csv(results, summary_csv)

    # Generate comparison plots
    if not args.no_plots and any(r.get("status") == "OK" for r in results):
        plots_path = os.path.join(args.output_dir, "comparison_plots.png")
        generate_comparison_plots(results, plots_path)

    # Top 3
    ok_results = [r for r in results if r.get("status") == "OK"]
    if ok_results:
        top3 = sorted(ok_results, key=lambda r: r.get("best_wer") or float("inf"))[:3]
        print()
        print("  Top 3 by Best WER:")
        for rank, r in enumerate(top3):
            wer_str = f"{r.get('best_wer', 0):.2f}%"
            print(f"    {rank+1}. {r['experiment_name']:<28s}  WER={wer_str}  "
                  f"({', '.join(r.get('enhancements', []))})")

    # Return non-zero exit code if any failed
    if fail_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
