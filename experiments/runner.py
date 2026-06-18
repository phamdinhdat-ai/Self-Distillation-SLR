"""
Experiment runner for SMKD enhancement suite.

Loads a YAML config, flattens it into a Trainer-compatible dict, runs
training, and saves results (checkpoint, history CSV, config copy) into a
timestamped run directory.

Usage:
    # Run baseline on synthetic data (fast smoke test)
    python experiments/runner.py --config configs/default.yaml --synthetic --epochs 6

    # Run with real data
    python experiments/runner.py --config configs/default.yaml --real

    # Override specific settings via CLI
    python experiments/runner.py --config configs/default.yaml --override loss.alpha=0.3
"""

import argparse
import os
import sys
import time
import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import yaml
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trainer import Trainer


# ---------------------------------------------------------------------------
# Config loading & flattening
# ---------------------------------------------------------------------------

def load_yaml_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def flatten_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flatten a nested YAML config into the flat dict expected by Trainer(cfg).

    Most keys are passed through as-is; nested sections are merged with
    section-qualified keys (e.g. model.d_model -> d_model) where the
    Trainer expects them.
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
    flat["cache_dir"] = data.get("cache_dir", None)

    # Temporal augmentation
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
    flat["save_stage_checkpoints"] = out.get("save_stage_checkpoints", True)
    flat["resume_from_stage"] = out.get("resume_from_stage", 0)
    flat["resume_ckpt_path"] = out.get("resume_ckpt_path", "")

    return flat


def override_config(flat: Dict[str, Any], overrides: list) -> Dict[str, Any]:
    """Apply CLI overrides like 'loss.alpha=0.3' to the flat config."""
    for ov in overrides:
        try:
            key, value = ov.split("=", 1)
            # Try to infer type
            if value.lower() == "true":
                value = True
            elif value.lower() == "false":
                value = False
            elif value.lower() == "none":
                value = None
            else:
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        pass  # keep as string
            flat[key] = value
        except ValueError:
            logger.warning(f"Invalid override: '{ov}' (expected key=value)")
    return flat


# ---------------------------------------------------------------------------
# Run directory & results saving
# ---------------------------------------------------------------------------

def setup_run_dir(experiment_name: str, timestamp: str = None) -> str:
    """Create runs/<experiment_name>/<timestamp>/ and return its path."""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("runs", experiment_name, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="SMKD Experiment Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", required=True, help="Path to YAML config file")
    p.add_argument("--override", nargs="*", default=[],
                   help="Override config values, e.g. loss.alpha=0.3")
    # Quick overrides for smoke testing
    p.add_argument("--synthetic", dest="synthetic_override", action="store_true", default=None)
    p.add_argument("--real", dest="synthetic_override", action="store_false")
    p.add_argument("--epochs", type=int, default=None, help="Override total epochs")
    p.add_argument("--d_model", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--no-plot", action="store_true", help="Skip saving WER/loss plot")
    return p.parse_args()


def main():
    args = parse_args()

    # Load config
    logger.info(f"Loading config: {args.config}")
    if not os.path.exists(args.config):
        # Try relative to configs/
        alt = os.path.join("configs", args.config)
        if os.path.exists(alt):
            args.config = alt
        elif os.path.exists(args.config + ".yaml"):
            args.config = args.config + ".yaml"
        else:
            raise FileNotFoundError(f"Config file not found: {args.config}")

    raw_cfg = load_yaml_config(args.config)
    config = flatten_config(raw_cfg)

    # Apply CLI quick overrides
    if args.synthetic_override is not None:
        config["synthetic"] = args.synthetic_override
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.d_model is not None:
        config["d_model"] = args.d_model
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size

    # Apply --override key=value pairs
    config = override_config(config, args.override)

    # Setup run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = setup_run_dir(config["experiment_name"], timestamp)
    config["ckpt_dir"] = os.path.join(run_dir, "checkpoints")

    # Save a copy of the effective config
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.dump(raw_cfg, f, default_flow_style=False, sort_keys=False)
    # Also save the flattened config for debugging
    with open(os.path.join(run_dir, "flat_config.yaml"), "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Experiment: {config['experiment_name']}")
    logger.debug("Config summary:")
    for k, v in sorted(config.items()):
        if not k.startswith("_"):
            logger.debug(f"  {k}: {v}")

    # Run training
    trainer = Trainer(config)
    trainer.train()

    # Save history CSV
    history_path = os.path.join(run_dir, "history.csv")
    save_history_csv(trainer.history, history_path)
    logger.info(f"History saved to: {history_path}")

    # Save WER/loss plot
    plot_path = os.path.join(run_dir, "wer_history.png")
    trainer.plot_history(save_path=plot_path, show=False)
    logger.info(f"Plot saved to: {plot_path}")

    # Print final summary
    print()
    print("=" * 70)
    logger.success(f"Experiment complete: {config['experiment_name']}")
    logger.info(f"Run directory: {run_dir}")
    if trainer.best_epoch is not None:
        logger.success(f"Best WER: {trainer.best_wer:.2f}% @ epoch {trainer.best_epoch}")
    print("=" * 70)

    return run_dir


if __name__ == "__main__":
    main()
