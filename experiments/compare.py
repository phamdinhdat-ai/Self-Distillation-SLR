"""
Compare results across multiple SMKD experiment runs.

Reads history.csv files from runs/*/ subdirectories and produces:
  1. A comparison table (best WER, final WER, convergence speed)
  2. An overlay WER plot
  3. An overlay loss plot

Usage:
    # Compare all runs under runs/
    python experiments/compare.py

    # Compare specific runs
    python experiments/compare.py runs/baseline/20250101_120000 runs/vac/20250101_130000

    # Output to file
    python experiments/compare.py --output comparison.png
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional


def find_run_dirs(base: str = "runs") -> List[str]:
    """Find all run directories (containing history.csv) under base."""
    run_dirs = []
    for root, dirs, files in os.walk(base):
        if "history.csv" in files:
            run_dirs.append(root)
    return sorted(run_dirs)


def load_history(path: str) -> Optional[Dict[str, List]]:
    """Load a history.csv file, return dict of column -> list of values."""
    try:
        with open(path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if not rows:
            return None
        columns = rows[0].keys()
        history = {col: [] for col in columns}
        for row in rows:
            for col in columns:
                val = row[col].strip()
                if val == "":
                    history[col].append(None)
                else:
                    try:
                        history[col].append(float(val))
                    except ValueError:
                        history[col].append(val)
        return history
    except Exception as e:
        print(f"  [!] Failed to load {path}: {e}")
        return None


def run_name(run_dir: str) -> str:
    """Derive a display name from the run directory path."""
    # runs/<experiment_name>/<timestamp>
    parts = Path(run_dir).parts
    if len(parts) >= 2:
        name = parts[-2]
        ts = parts[-1]
        # Shorten timestamp
        return f"{name}/{ts}"
    return run_dir


def build_summary(run_dirs: List[str]) -> List[Dict]:
    """Build a summary table row for each run."""
    summary = []
    for rd in run_dirs:
        hist = load_history(os.path.join(rd, "history.csv"))
        if hist is None:
            continue
        epochs = hist.get("epoch", [])
        wer_vals = [w for w in hist.get("wer", []) if w is not None]
        total_vals = [t for t in hist.get("total", []) if t is not None]

        best_wer = min(wer_vals) if wer_vals else float("inf")
        best_epoch = None
        if wer_vals:
            for e, w in zip(epochs, hist["wer"]):
                if w is not None and w == best_wer:
                    best_epoch = int(e)
                    break
        final_wer = wer_vals[-1] if wer_vals else None
        final_loss = total_vals[-1] if total_vals else None

        # Convergence: epoch at which WER reaches within 5% of best
        conv_epoch = None
        if wer_vals and best_wer < float("inf"):
            threshold = best_wer * 1.05
            for e, w in zip(epochs, hist["wer"]):
                if w is not None and w <= threshold:
                    conv_epoch = int(e)
                    break

        summary.append({
            "name": run_name(rd),
            "dir": rd,
            "epochs": len(epochs),
            "best_wer": best_wer,
            "best_epoch": best_epoch,
            "final_wer": final_wer,
            "final_loss": final_loss,
            "conv_epoch": conv_epoch,
            "history": hist,
        })
    return summary


def print_table(summary: List[Dict]):
    """Pretty-print comparison table."""
    print()
    print("=" * 95)
    print(f"{'Experiment':<30} {'Epochs':>6} {'Best WER':>10} {'@Ep':>5} {'Final WER':>10} {'Conv@Ep':>8}")
    print("-" * 95)
    for s in sorted(summary, key=lambda x: x["best_wer"]):
        name = s["name"][:29]
        best_wer = f"{s['best_wer']:.2f}%" if s["best_wer"] < float("inf") else "--"
        best_ep = f"{s['best_epoch']}" if s['best_epoch'] is not None else "--"
        final_wer = f"{s['final_wer']:.2f}%" if s['final_wer'] is not None else "--"
        conv = f"{s['conv_epoch']}" if s['conv_epoch'] is not None else "--"
        print(f"{name:<30} {s['epochs']:>6} {best_wer:>10} {best_ep:>5} {final_wer:>10} {conv:>8}")
    print("=" * 95)
    print()


def plot_comparison(summary: List[Dict], output_path: Optional[str] = None):
    """Generate overlay WER and loss plots."""
    import matplotlib.pyplot as plt

    n = len(summary)
    if n == 0:
        print("No runs to plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.cm.tab10(range(n))

    # --- WER plot ---
    ax = axes[0]
    for i, s in enumerate(summary):
        hist = s["history"]
        wer_epochs = [e for e, w in zip(hist["epoch"], hist["wer"]) if w is not None]
        wer_vals = [w for w in hist["wer"] if w is not None]
        if wer_vals:
            ax.plot(wer_epochs, wer_vals, marker=".", color=colors[i],
                    label=s["name"], linewidth=1.5, markersize=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("WER (%)")
    ax.set_title("WER over epochs")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.3)

    # --- Loss plot ---
    ax = axes[1]
    for i, s in enumerate(summary):
        hist = s["history"]
        total_epochs = [e for e, t in zip(hist["epoch"], hist["total"]) if t is not None]
        total_vals = [t for t in hist["total"] if t is not None]
        if total_vals:
            ax.plot(total_epochs, total_vals, color=colors[i],
                    label=s["name"], linewidth=1.5, alpha=0.85)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Total Loss")
    ax.set_title("Training loss over epochs")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.3)

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=120, bbox_inches="tight")
        print(f"Comparison plot saved to: {output_path}")
    else:
        plt.show()


def main():
    p = argparse.ArgumentParser(description="Compare SMKD experiment runs")
    p.add_argument("run_dirs", nargs="*", help="Run directories to compare (default: all under runs/)")
    p.add_argument("--output", "-o", default=None, help="Save comparison plot to path")
    args = p.parse_args()

    if args.run_dirs:
        run_dirs = args.run_dirs
    else:
        run_dirs = find_run_dirs("runs")

    if not run_dirs:
        print("No run directories found (no history.csv files under runs/).")
        print("Run an experiment first: python experiments/runner.py --config configs/default.yaml")
        return

    print(f"Found {len(run_dirs)} run(s):")
    for rd in run_dirs:
        print(f"  {rd}")

    summary = build_summary(run_dirs)
    if not summary:
        print("No valid history files found.")
        return

    print_table(summary)
    plot_comparison(summary, output_path=args.output)


if __name__ == "__main__":
    main()
