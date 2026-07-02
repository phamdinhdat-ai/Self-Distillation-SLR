"""
Plotting script for SMKD experiment results.
Parses individual/*.md result files and generates:
  1. WER comparison bar chart
  2. WER progression curves (all experiments)
  3. Loss curves (CTC(g), CTC(v), Seg, Total)
  4. Timing profile comparison
  5. Per-stage summary heatmap
  6. Training time vs WER scatter

Usage:
    python plot_experiments.py
    python plot_experiments.py --output-dir ./plots
"""
import re
import os
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------------------
# Parser — extracts structured data from individual .md result files
# ---------------------------------------------------------------------------

def parse_md_file(filepath: str) -> dict:
    """Parse a single experiment result markdown file into a dict."""
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    data = {"filename": Path(filepath).stem}

    # --- Overview table ---
    data["dev_wer"]   = _re_first(r"Dev WER\s*\|\s*\*?\*?([\d.]+)%", text, float)
    data["test_wer"]  = _re_first(r"Test WER\s*\|\s*\*?\*?([\d.]+)%", text, float)
    data["delta"]     = _re_first(r"Δ from baseline\s*\|\s*\*?\*?([+-][\d.]+)%", text, float)
    data["params_m"]   = _re_first(r"\|\s*Parameters\s*\|\s*([\d.]+)M", text, float)
    data["trainable_m"] = _re_first(r"\|\s*Trainable\s*\|\s*([\d.]+)M", text, float)
    data["total_time_h"] = _re_first(r"Est\. total time\s*\|\s*\*?\*?([\d.]+)h", text, float)
    data["category"]     = _re_first(r"\*\*Category:\*\*\s*(.+)", text, str)
    data["enhancements"] = _re_first(r"\|\s*Enhancements\s*\|\s*(.+?)\s*\|", text, str)

    # --- WER Progression table ---
    data["wer_prog"] = _parse_table(text, "WER Progression", cols=["epoch","stage","wer"], float_cols=[0,2], int_cols=[1])

    # --- Training Loss table ---
    data["loss_prog"] = _parse_table(text, "Training Loss Curves", cols=["epoch","stage","ctc_g","ctc_v","seg","total"], float_cols=[0,2,3,4,5], int_cols=[1])

    # --- Per-Stage Summary ---
    data["stage_summary"] = _parse_table(text, "Per-Stage Summary", cols=["stage","mode","epochs","ctc_g","ctc_v","seg","total"], float_cols=[3,4,5,6], int_cols=[0])

    # --- Timing Profile ---
    data["timing"] = _parse_table(text, "Timing Profile", cols=["phase","time_s","pct"], float_cols=[1,2], int_cols=[])
    # Strip 's' suffix from time values
    for row in data["timing"]:
        if isinstance(row.get("time_s"), str):
            row["time_s"] = float(row["time_s"].replace("s",""))

    return data


def _re_first(pattern: str, text: str, cast=float):
    m = re.search(pattern, text)
    if m:
        return cast(m.group(1))
    return None


def _parse_table(text: str, section: str, cols: list, float_cols: list, int_cols: list) -> list:
    """Parse a markdown table under a ## heading."""
    # Find section
    pattern = rf"##\s+.*?{section}.*?\n\n\|([\s\S]+?)(?=\n\n|\n##|\Z)"
    m = re.search(pattern, text)
    if not m:
        return []
    table_text = m.group(1)
    lines = [l.strip() for l in table_text.split("\n") if l.startswith("|") and not l.startswith("|---") and not l.startswith("| Epoch")]
    results = []
    for line in lines:
        cells = [c.strip() for c in line.split("|")[1:-1]]
        if not cells or not cells[0]:
            continue
        row = {}
        for i, col in enumerate(cols):
            if i >= len(cells):
                break
            val = cells[i].replace("%","").replace(",","")
            if val in ("—","-",""):
                row[col] = None
            elif i in float_cols:
                try:
                    row[col] = float(val)
                except ValueError:
                    row[col] = None
            elif i in int_cols:
                try:
                    row[col] = int(val)
                except ValueError:
                    row[col] = val
            else:
                row[col] = val
        if row:
            results.append(row)
    return results


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------

FIGSIZE = (14, 7)
COLORS = {
    "baseline":         "#444444",
    "loss_vac_entropy": "#2196F3",
    "model_pretrained": "#4CAF50",
    "model_transformer":"#FF9800",
    "train_amp":        "#9C27B0",
    "combined_tier1":   "#00BCD4",
    "combined_tier2":   "#F44336",
}
CATEGORY_COLORS = {
    "Reference":         "#757575",
    "Loss Enhancement":  "#2196F3",
    "Model Enhancement": "#4CAF50",
    "Training Method":   "#9C27B0",
    "Combined":          "#F44336",
}
LABELS = {
    "baseline":         "Baseline",
    "loss_vac_entropy": "VAC+Entropy",
    "model_pretrained": "Pretrained BB",
    "model_transformer":"Transformer",
    "train_amp":        "AMP (FP16)",
    "combined_tier1":   "Combined T1",
    "combined_tier2":   "Combined T2",
}


def plot_wer_bars(experiments: dict, output_dir: str):
    """Bar chart: Dev/Test WER sorted by Dev WER."""
    sorted_exps = sorted(experiments.items(), key=lambda x: x[1]["dev_wer"] or 99)
    names = [LABELS.get(k, k) for k, _ in sorted_exps]
    devs  = [v["dev_wer"] for _, v in sorted_exps]
    tests = [v["test_wer"] for _, v in sorted_exps]
    colors = [COLORS.get(k, "#888888") for k, _ in sorted_exps]

    x = np.arange(len(names))
    w = 0.35

    fig, ax = plt.subplots(figsize=FIGSIZE)
    bars1 = ax.bar(x - w/2, devs, w, label="Dev WER", color=[_lighten(c, 0.7) for c in colors], edgecolor=colors, linewidth=1.5)
    bars2 = ax.bar(x + w/2, tests, w, label="Test WER", color=[_lighten(c, 0.3) for c in colors], edgecolor=colors, linewidth=1.5)

    ax.set_ylabel("WER (%)")
    ax.set_title("SMKD Experiments — Dev & Test WER Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    ax.set_ylim(18, 24)

    for bar, val in zip(bars1, devs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15, f"{val:.1f}%", ha="center", va="bottom", fontsize=7)
    for bar, val in zip(bars2, tests):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15, f"{val:.1f}%", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "wer_bars.png"), dpi=150)
    plt.close(fig)
    print(f"  ✓ wer_bars.png")


def plot_wer_progression(experiments: dict, output_dir: str):
    """Line chart: WER vs epoch for all experiments, with stage background shading."""
    fig, ax = plt.subplots(figsize=FIGSIZE)

    # Stage shading
    ax.axvspan(0, 30, alpha=0.06, color="blue", label="Stage 1 (Sync)")
    ax.axvspan(30, 90, alpha=0.06, color="orange", label="Stage 2 (GSBA)")
    ax.axvspan(90, 100, alpha=0.06, color="green", label="Stage 3 (Decouple)")

    for exp_name, exp_data in experiments.items():
        prog = exp_data.get("wer_prog", [])
        if not prog:
            continue
        epochs = [p["epoch"] for p in prog]
        wers   = [p["wer"] for p in prog]
        color = COLORS.get(exp_name, "#888888")
        label = LABELS.get(exp_name, exp_name)
        lw = 2.5 if "combined" in exp_name else 1.5
        ls = "--" if exp_name == "baseline" else "-"
        ax.plot(epochs, wers, color=color, linewidth=lw, linestyle=ls, label=label, marker="o", markersize=3)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Dev WER (%)")
    ax.set_title("WER Progression Across Experiments")
    ax.legend(fontsize=8, ncol=2)
    ax.set_xlim(0, 105)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "wer_progression.png"), dpi=150)
    plt.close(fig)
    print(f"  ✓ wer_progression.png")


def plot_loss_curves(experiments: dict, output_dir: str):
    """2×2 grid: CTC(g), CTC(v), Seg, Total loss curves."""
    metrics = [
        ("ctc_g", "CTC(g) Loss"),
        ("ctc_v", "CTC(v) Loss"),
        ("seg",   "Segmentation CE Loss"),
        ("total", "Total Loss"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    for (key, title), ax in zip(metrics, axes.flat):
        for exp_name, exp_data in experiments.items():
            prog = exp_data.get("loss_prog", [])
            if not prog:
                continue
            epochs = [p["epoch"] for p in prog]
            vals   = [p.get(key) for p in prog]
            if all(v is None for v in vals):
                continue
            color = COLORS.get(exp_name, "#888888")
            label = LABELS.get(exp_name, exp_name)
            lw = 2.0 if "combined" in exp_name else 1.2
            ls = "--" if exp_name == "baseline" else "-"
            ax.plot(epochs, vals, color=color, linewidth=lw, linestyle=ls, label=label)

        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(fontsize=6, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 105)

    fig.suptitle("Training Loss Curves — All Experiments", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "loss_curves.png"), dpi=150)
    plt.close(fig)
    print(f"  ✓ loss_curves.png")


def plot_timing(experiments: dict, output_dir: str):
    """Stacked bar: timing breakdown per experiment."""
    phases = ["Data transfer", "Forward pass", "GSBA", "Loss compute", "Backward+Opt"]
    phase_keys = {"Data transfer": "data_xfer", "Forward pass": "forward", "GSBA": "gsba", "Loss compute": "loss", "Backward+Opt": "backward"}

    exps_with_timing = {}
    for name, exp in experiments.items():
        timing = exp.get("timing", [])
        if not timing:
            continue
        tmap = {}
        for row in timing:
            phase = row.get("phase","")
            t = row.get("time_s", 0)
            if t is None:
                t = 0
            tmap[phase] = t
        total = sum(tmap.values())
        exps_with_timing[name] = {"total": total, "times": {k: tmap.get(k, 0) for k in phases}}

    if not exps_with_timing:
        return

    names = [LABELS.get(k, k) for k in exps_with_timing]
    colors_phase = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0", "#F44336"]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    x = np.arange(len(names))
    bottom = np.zeros(len(names))
    for i, phase in enumerate(phases):
        vals = [exps_with_timing[n]["times"][phase] for n in exps_with_timing]
        ax.barh(x, vals, left=bottom, color=colors_phase[i], label=phase, edgecolor="white", linewidth=0.5)
        bottom += vals

    ax.set_yticks(x)
    ax.set_yticklabels(names)
    ax.set_xlabel("Time per epoch (seconds)")
    ax.set_title("Timing Breakdown per Experiment")
    ax.legend(loc="lower right", fontsize=8)
    ax.invert_yaxis()

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "timing.png"), dpi=150)
    plt.close(fig)
    print(f"  ✓ timing.png")


def plot_time_vs_wer(experiments: dict, output_dir: str):
    """Scatter: training time vs Dev WER."""
    fig, ax = plt.subplots(figsize=(10, 7))

    for exp_name, exp_data in experiments.items():
        t = exp_data.get("total_time_h")
        w = exp_data.get("dev_wer")
        if t is None or w is None:
            continue
        color = COLORS.get(exp_name, "#888888")
        label = LABELS.get(exp_name, exp_name)
        size = 200 if "combined" in exp_name else 120
        marker = "*" if "combined" in exp_name else "o"
        ax.scatter(t, w, c=color, s=size, marker=marker, label=label,
                   edgecolors="black", linewidth=0.5, zorder=5)
        ax.annotate(label, (t, w), textcoords="offset points", xytext=(8, 4), fontsize=7)

    ax.set_xlabel("Training Time (hours)")
    ax.set_ylabel("Dev WER (%)")
    ax.set_title("Training Time vs Dev WER")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "time_vs_wer.png"), dpi=150)
    plt.close(fig)
    print(f"  ✓ time_vs_wer.png")


def plot_stage_summary(experiments: dict, output_dir: str):
    """Grouped bar: per-stage average CTC(g) loss for each experiment."""
    stages = ["Stage 1", "Stage 2", "Stage 3"]
    fig, ax = plt.subplots(figsize=FIGSIZE)

    sorted_exps = sorted(experiments.items(), key=lambda x: x[1]["dev_wer"] or 99)
    names = [LABELS.get(k, k) for k, _ in sorted_exps]
    colors = [COLORS.get(k, "#888888") for k, _ in sorted_exps]

    x = np.arange(len(names))
    w = 0.25

    for i, (stage_label, stage_num) in enumerate(zip(stages, [1, 2, 3])):
        vals = []
        for _, exp in sorted_exps:
            summary = exp.get("stage_summary", [])
            found = None
            for s in summary:
                if s.get("stage") == stage_num:
                    found = s.get("ctc_g")
                    break
            vals.append(found if found is not None else 0)
        offset = (i - 1) * w
        ax.bar(x + offset, vals, w, label=stage_label, alpha=0.85)

    ax.set_ylabel("Avg CTC(g) Loss")
    ax.set_title("Per-Stage CTC(g) Loss Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "stage_summary.png"), dpi=150)
    plt.close(fig)
    print(f"  ✓ stage_summary.png")


def plot_delta_waterfall(experiments: dict, output_dir: str):
    """Waterfall chart: WER delta from baseline."""
    baseline_wer = experiments.get("baseline", {}).get("dev_wer", 21.88)
    sorted_exps = sorted(
        [(k,v) for k,v in experiments.items() if k != "baseline"],
        key=lambda x: x[1].get("dev_wer", 99)
    )

    names = [LABELS.get(k, k) for k, _ in sorted_exps]
    deltas = [(v.get("dev_wer", baseline_wer) - baseline_wer) for _, v in sorted_exps]
    colors_bar = ["#4CAF50" if d < 0 else "#F44336" for d in deltas]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(names, deltas, color=colors_bar, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Δ Dev WER from Baseline (%)")
    ax.set_title("WER Improvement Over Baseline")
    ax.invert_yaxis()

    for bar, d in zip(bars, deltas):
        sign = "↓" if d < 0 else "↑"
        ax.text(bar.get_width() - 0.08 if d < 0 else bar.get_width() + 0.02,
                bar.get_y() + bar.get_height()/2,
                f"{sign}{abs(d):.2f}%", va="center", ha="right" if d < 0 else "left",
                fontsize=9, fontweight="bold",
                color="white" if abs(d) > 0.5 else "black")

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "delta_waterfall.png"), dpi=150)
    plt.close(fig)
    print(f"  ✓ delta_waterfall.png")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _lighten(hex_color: str, factor: float) -> str:
    """Lighten a hex color by factor (0=black, 1=original, >1=lighter)."""
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    r = min(255, int(r + (255 - r) * (1 - factor)))
    g = min(255, int(g + (255 - g) * (1 - factor)))
    b = min(255, int(b + (255 - b) * (1 - factor)))
    return f"#{r:02x}{g:02x}{b:02x}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Plot SMKD experiment results")
    parser.add_argument("--results-dir", default="runs/results/individual",
                        help="Directory containing individual .md result files")
    parser.add_argument("--output-dir", default="runs/results/plots",
                        help="Directory to save plots")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse all .md files
    experiments = {}
    for fpath in sorted(results_dir.glob("*.md")):
        if fpath.name == "INDEX.md":
            continue
        data = parse_md_file(str(fpath))
        experiments[data["filename"]] = data
        print(f"  Parsed: {data['filename']} (Dev WER: {data.get('dev_wer','?')}%)")

    if not experiments:
        print("No experiment files found!")
        return

    print(f"\nGenerating plots in {output_dir}/ ...\n")

    plot_wer_bars(experiments, str(output_dir))
    plot_wer_progression(experiments, str(output_dir))
    plot_loss_curves(experiments, str(output_dir))
    plot_timing(experiments, str(output_dir))
    plot_time_vs_wer(experiments, str(output_dir))
    plot_stage_summary(experiments, str(output_dir))
    plot_delta_waterfall(experiments, str(output_dir))

    print(f"\nDone! {len(os.listdir(output_dir))} plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
