#!/usr/bin/env python3
"""
PHOENIX-2014T Dataset Analysis
==============================
Comprehensive statistical analysis and visualization of the PHOENIX-2014T
German Sign Language dataset used for Continuous Sign Language Recognition (CSLR).

Analyzes:
  1. Split sizes & sample counts
  2. Frame count distributions per split
  3. Gloss sequence length distributions
  4. Vocabulary statistics & gloss frequency (top/bottom N)
  5. Speaker distribution across splits
  6. Translation (German text) length & word frequency
  7. Train/dev/test gloss overlap
  8. Memory & throughput estimates for training

Outputs:
  - Console summary with key statistics
  - plots/ directory with PNG visualizations (if matplotlib available)
  - analysis_report.csv with per-sample details (optional)

Usage:
    python experiments/analyze_data.py
    python experiments/analyze_data.py --data_root ./data/phoenix
    python experiments/analyze_data.py --data_root ./data/phoenix --output report.csv --plots
    python experiments/analyze_data.py --no-plots              # text-only
"""

import argparse
import csv
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Allow importing from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================================
# Constants — PHOENIX-2014T known statistics
# ============================================================================

PHOENIX_KNOWN = {
    "train_samples": 7096,
    "dev_samples": 519,
    "test_samples": 642,
    "vocab_size": 1066,       # glosses (paper-reported; corpus may differ slightly)
    "num_signers": 9,
    "fps": 25,
    "avg_frames_raw": 110,    # before subsampling
}

# ============================================================================
# CSV parsing helpers
# ============================================================================

def find_corpus_csv(data_root: str, split: str) -> Optional[str]:
    """Locate the PHOENIX-2014-T corpus CSV for a given split."""
    import glob
    patterns = [
        os.path.join(data_root, "*", f"PHOENIX-2014-T.{split}.corpus.csv"),
        os.path.join(data_root, f"PHOENIX-2014-T.{split}.corpus.csv"),
        os.path.join(data_root, "*", f"*.{split}.corpus.csv"),
        os.path.join(data_root, f"*.{split}.corpus.csv"),
        os.path.join(data_root, "**", f"*{split}*corpus*.csv"),
    ]
    for pattern in patterns:
        matches = sorted(glob.glob(pattern, recursive=True))
        if matches:
            return matches[0]
    return None


def load_csv(path: str) -> List[Dict[str, str]]:
    """Load a pipe-separated PHOENIX corpus CSV."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="|")
        for row in reader:
            rows.append(row)
    return rows


def count_frames_in_dir(sample_dir: str) -> int:
    """Count image frames in a sample directory (png/jpg)."""
    import glob
    if not os.path.isdir(sample_dir):
        return 0
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        paths = glob.glob(os.path.join(sample_dir, ext))
        if paths:
            return len(paths)
    return 0


# ============================================================================
# Analysis functions
# ============================================================================

def analyze_splits(data_root: str) -> Dict[str, Dict]:
    """
    Parse all three corpus CSVs and return per-split statistics.
    """
    results = {}
    for split in ("train", "dev", "test"):
        csv_path = find_corpus_csv(data_root, split)
        if csv_path is None:
            print(f"  [!] No corpus CSV found for '{split}'. Skipping.")
            results[split] = {"samples": 0, "csv_path": None}
            continue

        rows = load_csv(csv_path)
        results[split] = {
            "csv_path": csv_path,
            "samples": len(rows),
            "rows": rows,
        }
    return results


def analyze_frames(data_root: str, rows: List[Dict], split: str) -> Dict:
    """
    Analyze frame counts per sample.
    Scans actual frame directories if they exist, otherwise estimates from CSV.
    """
    split_dir = os.path.join(data_root, split)
    exists = os.path.isdir(split_dir)

    frame_counts = []
    missing_dirs = 0
    min_frames = float("inf")
    max_frames = 0

    for row in rows:
        name = row.get("name", "").strip()
        if not name:
            video = row.get("video", "")
            name = video.split("/")[0] if video else ""
        if not name:
            continue

        if exists:
            n = count_frames_in_dir(os.path.join(split_dir, name))
            if n == 0:
                missing_dirs += 1
                # Estimate from FPS * duration if start/end present
                try:
                    start = float(row.get("start", 0))
                    end = float(row.get("end", 0))
                    duration = max(0, end - start)
                    n = max(1, int(duration * 25))  # PHOENIX is 25 fps
                except (ValueError, TypeError):
                    n = 100  # fallback
        else:
            # Estimate from CSV start/end timestamps
            try:
                start = float(row.get("start", 0))
                end = float(row.get("end", 0))
                duration = max(0, end - start)
                n = max(1, int(duration * 25))
            except (ValueError, TypeError):
                n = 0

        frame_counts.append(n)
        if n > 0:
            min_frames = min(min_frames, n)
            max_frames = max(max_frames, n)

    if not frame_counts:
        return {"count": 0, "avg": 0, "min": 0, "max": 0, "total": 0}

    total = sum(frame_counts)
    avg = total / len(frame_counts)

    # Percentiles
    sorted_counts = sorted(frame_counts)
    p25 = sorted_counts[int(len(sorted_counts) * 0.25)]
    p50 = sorted_counts[int(len(sorted_counts) * 0.50)]
    p75 = sorted_counts[int(len(sorted_counts) * 0.75)]
    p90 = sorted_counts[int(len(sorted_counts) * 0.90)]
    p95 = sorted_counts[int(len(sorted_counts) * 0.95)]
    p99 = sorted_counts[int(len(sorted_counts) * 0.99)]

    return {
        "count": len(frame_counts),
        "avg": avg,
        "min": min_frames,
        "max": max_frames,
        "total": total,
        "p25": p25, "p50": p50, "p75": p75, "p90": p90, "p95": p95, "p99": p99,
        "missing_dirs": missing_dirs,
        "all_counts": frame_counts,
    }


def analyze_glosses(rows: List[Dict]) -> Dict:
    """Analyze gloss sequences: lengths, vocabulary, frequency."""
    gloss_sequences = []
    all_glosses = []
    unique_glosses = set()
    lengths = []

    for row in rows:
        orth = row.get("orth", "").strip()
        glosses = orth.split() if orth else []
        gloss_sequences.append(glosses)
        all_glosses.extend(glosses)
        unique_glosses.update(glosses)
        lengths.append(len(glosses))

    if not lengths:
        return {"vocab_size": 0, "total_glosses": 0, "avg_len": 0}

    freq = Counter(all_glosses)
    sorted_lengths = sorted(lengths)

    return {
        "vocab_size": len(unique_glosses),
        "total_glosses": len(all_glosses),
        "total_tokens": sum(lengths),
        "avg_len": sum(lengths) / len(lengths),
        "min_len": min(lengths),
        "max_len": max(lengths),
        "p50": sorted_lengths[len(sorted_lengths) // 2],
        "p90": sorted_lengths[int(len(sorted_lengths) * 0.9)],
        "p95": sorted_lengths[int(len(sorted_lengths) * 0.95)],
        "freq_top20": freq.most_common(20),
        "freq_bottom20": freq.most_common()[-20:],
        "all_lengths": lengths,
        "all_glosses": all_glosses,
    }


def analyze_translations(rows: List[Dict]) -> Dict:
    """Analyze German translation text."""
    translations = []
    word_counter = Counter()
    lengths = []

    for row in rows:
        text = row.get("translation", "").strip()
        if text:
            translations.append(text)
            words = text.split()
            lengths.append(len(words))
            word_counter.update(words)

    if not lengths:
        return {"count": 0, "avg_len": 0}

    sorted_lengths = sorted(lengths)
    return {
        "count": len(translations),
        "total_words": sum(lengths),
        "avg_len": sum(lengths) / len(lengths),
        "min_len": min(lengths),
        "max_len": max(lengths),
        "p50": sorted_lengths[len(sorted_lengths) // 2],
        "p95": sorted_lengths[int(len(sorted_lengths) * 0.95)],
        "vocab_size": len(word_counter),
        "top20_words": word_counter.most_common(20),
        "all_lengths": lengths,
    }


def analyze_speakers(rows: List[Dict]) -> Dict:
    """Analyze speaker distribution."""
    speakers = Counter()
    for row in rows:
        speaker = row.get("speaker", "").strip()
        if speaker:
            speakers[speaker] += 1
    return {
        "num_speakers": len(speakers),
        "distribution": speakers.most_common(),
    }


def gloss_overlap(split_data: Dict[str, Dict]) -> Dict:
    """
    Compute gloss vocabulary overlap between train/dev/test.
    """
    vocabs = {}
    for split in ("train", "dev", "test"):
        data = split_data.get(split, {})
        rows = data.get("rows", [])
        glosses = set()
        for row in rows:
            orth = row.get("orth", "").strip()
            if orth:
                glosses.update(orth.split())
        vocabs[split] = glosses

    results = {}
    for s1 in ("train", "dev", "test"):
        for s2 in ("dev", "test", "train"):
            if s1 >= s2:
                continue
            v1, v2 = vocabs.get(s1, set()), vocabs.get(s2, set())
            both = v1 & v2
            only_s1 = v1 - v2
            only_s2 = v2 - v1
            results[f"{s1}_vs_{s2}"] = {
                "shared": len(both),
                "only_in_s1": len(only_s1),
                "only_in_s2": len(only_s2),
                "jaccard": len(both) / len(v1 | v2) if (v1 | v2) else 0,
            }

    return results


def estimate_memory(avg_frames: float, batch_size: int, d_model: int = 512,
                    backbone: str = "resnet18", seq_len_out: int = 75,
                    vocab_size: int = 1088) -> Dict:
    """
    Estimate GPU memory requirements for training.
    """
    # Spatial feature map: (B*T', C_feat, H', W') -> pooled (B*T', 512)
    # VisualModule: backbone ~11M params, TemporalCNN ~2.6M
    # ContextualModule: BiLSTM ~10.5M
    # Classifier: (vocab+1) * d_model

    # Rough activation memory (FP32)
    T_prime = int(avg_frames / 2)  # after MaxPool1d stride=2
    feat_dim = 512

    # Backbone activations (approximate: batch * frames * 3 * 224 * 224 input)
    input_bytes = batch_size * int(avg_frames) * 3 * 224 * 224 * 4  # float32

    # Feature activations after backbone + temporal
    feature_bytes = batch_size * T_prime * feat_dim * 4

    # LSTM hidden states (B, T', 2 * hidden_size * num_layers) each direction
    lstm_bytes = batch_size * T_prime * 2 * 512 * 4 * 2  # bidirectional + c/h

    # Logits (B, T', vocab+1)
    logit_bytes = batch_size * T_prime * (vocab_size + 1) * 4

    # CTC loss path
    ctc_bytes = batch_size * T_prime * (vocab_size + 1) * 4

    total_act_mb = (input_bytes + feature_bytes + lstm_bytes + logit_bytes + ctc_bytes) / (1024**2)

    # Param memory
    params = {
        "resnet18": 11.7e6,
        "mobilenet_v3_small": 2.54e6,
        "efficientnet_b0": 5.29e6,
    }
    backbone_params_m = params.get(backbone, 11.7e6) / 1e6
    temporal_params_m = 2.62
    lstm_params_m = 10.5
    classifier_params_m = (vocab_size + 1) * feat_dim / 1e6

    total_params = backbone_params_m + temporal_params_m + lstm_params_m + classifier_params_m
    param_mem_mb = total_params * 4  # FP32
    optimizer_mem_mb = param_mem_mb * 2  # Adam (m + v)

    total_mem_mb = param_mem_mb + optimizer_mem_mb + total_act_mb

    return {
        "total_params_M": round(total_params, 1),
        "param_memory_MB": round(param_mem_mb, 0),
        "optimizer_memory_MB": round(optimizer_mem_mb, 0),
        "activation_memory_MB": round(total_act_mb, 0),
        "total_estimated_MB": round(total_mem_mb, 0),
        "total_estimated_GB": round(total_mem_mb / 1024, 2),
        "avg_frames_input": int(avg_frames),
        "avg_frames_output": T_prime,
        "batch_size": batch_size,
    }


def format_time(seconds: float) -> str:
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        return f"{seconds / 3600:.1f}h"


# ============================================================================
# Plotting
# ============================================================================

def make_plots(split_data: Dict[str, Dict], frame_stats: Dict[str, Dict],
               gloss_stats: Dict[str, Dict], trans_stats: Dict[str, Dict],
               out_dir: str = "plots"):
    """Generate all plots and save to out_dir."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("\n[!] matplotlib not available. Skipping plots.")
        return

    os.makedirs(out_dir, exist_ok=True)
    colors = {"train": "#2196F3", "dev": "#4CAF50", "test": "#FF9800"}

    # ------------------------------------------------------------------
    # 1. Frame count histogram (per split, overlaid)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, split in zip(axes, ("train", "dev", "test")):
        counts = frame_stats.get(split, {}).get("all_counts", [])
        if not counts:
            ax.set_title(f"{split} (no data)")
            continue
        ax.hist(counts, bins=40, color=colors[split], alpha=0.7, edgecolor="white")
        avg = frame_stats[split]["avg"]
        ax.axvline(avg, color="red", linestyle="--", linewidth=1.5,
                   label=f"avg={avg:.0f}")
        ax.set_title(f"{split} ({len(counts)} samples)")
        ax.set_xlabel("Frames")
        ax.set_ylabel("Samples")
        ax.legend(fontsize=8)
    fig.suptitle("Frame Count Distribution per Split (PHOENIX-2014T)",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "frame_distribution.png"), dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 2. Gloss sequence length histogram
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, split in zip(axes, ("train", "dev", "test")):
        lengths = gloss_stats.get(split, {}).get("all_lengths", [])
        if not lengths:
            ax.set_title(f"{split} (no data)")
            continue
        ax.hist(lengths, bins=25, color=colors[split], alpha=0.7, edgecolor="white")
        avg = gloss_stats[split]["avg_len"]
        ax.axvline(avg, color="red", linestyle="--", linewidth=1.5,
                   label=f"avg={avg:.1f}")
        ax.set_title(f"{split} ({len(lengths)} samples)")
        ax.set_xlabel("Glosses per sentence")
        ax.set_ylabel("Samples")
        ax.legend(fontsize=8)
    fig.suptitle("Gloss Sequence Length Distribution", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "gloss_length_distribution.png"), dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 3. Top-30 gloss frequency (train)
    # ------------------------------------------------------------------
    train_gloss = gloss_stats.get("train", {})
    freq = train_gloss.get("freq_top20", [])[:30]
    if freq:
        fig, ax = plt.subplots(figsize=(12, 5))
        names, counts = zip(*freq)
        ypos = range(len(names))
        ax.barh(ypos, counts, color="#2196F3", alpha=0.8)
        ax.set_yticks(ypos)
        ax.set_yticklabels(names, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("Frequency")
        ax.set_title("Top-30 Most Frequent Glosses (train)", fontweight="bold")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "top_glosses.png"), dpi=150)
        plt.close(fig)

    # ------------------------------------------------------------------
    # 4. Speaker distribution (train)
    # ------------------------------------------------------------------
    speakers = split_data.get("train", {}).get("speakers", {})
    dist = speakers.get("distribution", [])
    if dist:
        fig, ax = plt.subplots(figsize=(8, 4))
        names, counts = zip(*dist)
        bars = ax.bar(range(len(names)), counts, color="#2196F3", alpha=0.8)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, fontsize=8)
        ax.set_ylabel("Samples")
        ax.set_title("Samples per Signer (train)", fontweight="bold")
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                    str(count), ha="center", fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "speaker_distribution.png"), dpi=150)
        plt.close(fig)

    # ------------------------------------------------------------------
    # 5. Combined summary dashboard
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Row 1: Frame count boxplot, Gloss length boxplot, Train gloss freq
    frame_data = []
    frame_labels = []
    for split in ("train", "dev", "test"):
        c = frame_stats.get(split, {}).get("all_counts", [])
        if c:
            frame_data.append(c)
            frame_labels.append(split)
    if frame_data:
        axes[0, 0].boxplot(frame_data, labels=frame_labels, patch_artist=True,
                           boxprops=dict(facecolor="lightblue"))
        axes[0, 0].set_title("Frame Counts per Split")
        axes[0, 0].set_ylabel("Frames")

    gloss_len_data = []
    gloss_labels = []
    for split in ("train", "dev", "test"):
        gl = gloss_stats.get(split, {}).get("all_lengths", [])
        if gl:
            gloss_len_data.append(gl)
            gloss_labels.append(split)
    if gloss_len_data:
        axes[0, 1].boxplot(gloss_len_data, labels=gloss_labels, patch_artist=True,
                           boxprops=dict(facecolor="lightgreen"))
        axes[0, 1].set_title("Gloss Sequence Lengths")
        axes[0, 1].set_ylabel("Glosses")

    if freq:
        top10_names, top10_counts = zip(*freq[:10])
        axes[0, 2].barh(range(len(top10_names)), top10_counts, color="#2196F3", alpha=0.8)
        axes[0, 2].set_yticks(range(len(top10_names)))
        axes[0, 2].set_yticklabels(top10_names, fontsize=7)
        axes[0, 2].invert_yaxis()
        axes[0, 2].set_title("Top-10 Glosses (train)")

    # Row 2: Translation length, Frames vs Glosses scatter, Vocabulary overlap
    trans_len_data = []
    trans_labels = []
    for split in ("train", "dev", "test"):
        tl = trans_stats.get(split, {}).get("all_lengths", [])
        if tl:
            trans_len_data.append(tl)
            trans_labels.append(split)
    if trans_len_data:
        axes[1, 0].boxplot(trans_len_data, labels=trans_labels, patch_artist=True,
                           boxprops=dict(facecolor="lightcoral"))
        axes[1, 0].set_title("Translation Word Counts")
        axes[1, 0].set_ylabel("Words")

    # Frames vs glosses scatter (train only)
    train_frames = frame_stats.get("train", {}).get("all_counts", [])
    train_glosses = gloss_stats.get("train", {}).get("all_lengths", [])
    if train_frames and train_glosses and len(train_frames) == len(train_glosses):
        axes[1, 1].scatter(train_frames, train_glosses, alpha=0.3, s=4,
                           color="#2196F3")
        axes[1, 1].set_xlabel("Frames")
        axes[1, 1].set_ylabel("Glosses")
        axes[1, 1].set_title("Frames vs Gloss Length (train)")

    # Vocabulary overlap Venn-style bar chart
    overlap = gloss_overlap(split_data)
    pairs = [("train_vs_dev", "Train ∩ Dev"), ("train_vs_test", "Train ∩ Test"),
             ("dev_vs_test", "Dev ∩ Test")]
    pair_data = []
    for key, label in pairs:
        if key in overlap:
            o = overlap[key]
            pair_data.append((label, o["shared"], o["only_in_s1"], o["only_in_s2"]))
    if pair_data:
        x = np.arange(len(pair_data))
        w = 0.25
        axes[1, 2].bar(x - w, [p[1] for p in pair_data], w, label="Shared", color="green", alpha=0.7)
        axes[1, 2].bar(x, [p[2] for p in pair_data], w, label="Only A", color="blue", alpha=0.5)
        axes[1, 2].bar(x + w, [p[3] for p in pair_data], w, label="Only B", color="orange", alpha=0.5)
        axes[1, 2].set_xticks(x)
        axes[1, 2].set_xticklabels([p[0] for p in pair_data], fontsize=8)
        axes[1, 2].set_title("Gloss Vocabulary Overlap")
        axes[1, 2].legend(fontsize=7)

    fig.suptitle("PHOENIX-2014T Dataset Dashboard", fontweight="bold", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "dashboard.png"), dpi=150)
    plt.close(fig)

    print(f"\n  Plots saved to: {os.path.abspath(out_dir)}/")


# ============================================================================
# Report output
# ============================================================================

def print_report(split_data: Dict[str, Dict], frame_stats: Dict[str, Dict],
                 gloss_stats: Dict[str, Dict], trans_stats: Dict[str, Dict],
                 mem_est: Dict, args):
    """Print a formatted analysis report to console."""
    sep = "=" * 72
    sub = "-" * 72

    print(f"\n{sep}")
    print("  PHOENIX-2014T Dataset Analysis Report")
    print(sep)

    # ---- 1. Overview ----
    print(f"\n{' Dataset Overview ':-^72}")
    print(f"  Data root       : {args.data_root}")
    for split in ("train", "dev", "test"):
        data = split_data.get(split, {})
        csv_path = data.get("csv_path")
        n = data.get("samples", 0)
        status = f"✓ {csv_path}" if csv_path else "✗ not found"
        print(f"  {split:>5s} samples : {n:>6d}  ({status})")

    # ---- 2. Frame statistics ----
    print(f"\n{' Frame Statistics ':-^72}")
    print(f"  {'Split':>6s}  {'Samples':>8s}  {'Avg':>8s}  {'Median':>8s}  "
          f"{'Min':>6s}  {'Max':>6s}  {'P90':>6s}  {'P95':>6s}  {'P99':>6s}")
    print(f"  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*8}  "
          f"{'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}")
    for split in ("train", "dev", "test"):
        fs = frame_stats.get(split, {})
        if fs.get("count", 0) == 0:
            continue
        print(f"  {split:>6s}  {fs['count']:>8d}  {fs['avg']:>8.1f}  "
              f"{fs['p50']:>8d}  {fs['min']:>6d}  {fs['max']:>6d}  "
              f"{fs['p90']:>6d}  {fs['p95']:>6d}  {fs['p99']:>6d}")

    # ---- 3. Gloss statistics ----
    print(f"\n{' Gloss Statistics ':-^72}")
    print(f"  {'Split':>6s}  {'Vocab':>8s}  {'Tokens':>10s}  {'Avg':>8s}  "
          f"{'Median':>8s}  {'Min':>6s}  {'Max':>6s}  {'P95':>6s}")
    print(f"  {'-'*6}  {'-'*8}  {'-'*10}  {'-'*8}  "
          f"{'-'*8}  {'-'*6}  {'-'*6}  {'-'*6}")
    for split in ("train", "dev", "test"):
        gs = gloss_stats.get(split, {})
        if gs.get("vocab_size", 0) == 0:
            continue
        print(f"  {split:>6s}  {gs['vocab_size']:>8d}  {gs['total_tokens']:>10d}  "
              f"{gs['avg_len']:>8.2f}  {gs['p50']:>8d}  "
              f"{gs['min_len']:>6d}  {gs['max_len']:>6d}  {gs['p95']:>6d}")

    # Top glosses
    train_gloss = gloss_stats.get("train", {})
    top = train_gloss.get("freq_top20", [])[:10]
    if top:
        print(f"\n  Top-10 train glosses:")
        for i, (gloss, count) in enumerate(top, 1):
            pct = count / max(train_gloss.get("total_tokens", 1), 1) * 100
            print(f"    {i:>2d}. {gloss:<16s} {count:>6d} ({pct:.1f}%)")

    # Rare glosses
    bottom = train_gloss.get("freq_bottom20", [])[:5]
    if bottom:
        print(f"\n  Rarest train glosses (each appears once):")
        for gloss, count in bottom:
            print(f"      {gloss:<16s} {count}")

    # ---- 4. Translation statistics ----
    print(f"\n{' Translation (German) Statistics ':-^72}")
    for split in ("train", "dev", "test"):
        ts = trans_stats.get(split, {})
        if ts.get("count", 0) == 0:
            continue
        print(f"  {split:>5s}: {ts['count']} sentences, {ts['total_words']} words, "
              f"{ts['vocab_size']} unique words, "
              f"avg={ts['avg_len']:.1f} words, "
              f"median={ts['p50']}, max={ts['max_len']}")

    # Top translation words
    train_trans = trans_stats.get("train", {})
    top_words = train_trans.get("top20_words", [])[:10]
    if top_words:
        print(f"\n  Top-10 German words (train):")
        for i, (word, count) in enumerate(top_words, 1):
            print(f"    {i:>2d}. {word:<16s} {count:>6d}")

    # ---- 5. Speaker distribution ----
    speakers = split_data.get("train", {}).get("speakers", {})
    dist = speakers.get("distribution", [])
    if dist:
        print(f"\n{' Signer Distribution (train) ':-^72}")
        total_samples = split_data["train"]["samples"]
        for speaker, count in dist:
            pct = count / max(total_samples, 1) * 100
            bar = "█" * int(pct / 2)
            print(f"  {speaker:<12s} {count:>6d} ({pct:>5.1f}%)  {bar}")

    # ---- 6. Vocabulary overlap ----
    overlap = gloss_overlap(split_data)
    print(f"\n{' Gloss Vocabulary Overlap ':-^72}")
    pairs = [("train_vs_dev", "Train ↔ Dev"), ("train_vs_test", "Train ↔ Test"),
             ("dev_vs_test", "Dev ↔ Test")]
    for key, label in pairs:
        if key in overlap:
            o = overlap[key]
            v1 = o["shared"] + o["only_in_s1"]
            v2 = o["shared"] + o["only_in_s2"]
            print(f"  {label:<16s}  shared={o['shared']:>4d}  "
                  f"only_A={o['only_in_s1']:>4d}  only_B={o['only_in_s2']:>4d}  "
                  f"Jaccard={o['jaccard']:.3f}  "
                  f"(|A|={v1}, |B|={v2})")

    # ---- 7. Training estimates ----
    print(f"\n{' Training Throughput Estimates ':-^72}")
    print(f"  Backbone          : {args.backbone}")
    print(f"  Batch size        : {args.batch_size}")
    print(f"  d_model           : {args.d_model}")
    print(f"  Total params      : {mem_est['total_params_M']:.1f} M")
    print(f"  Estimated GPU mem : {mem_est['total_estimated_MB']:.0f} MB "
          f"({mem_est['total_estimated_GB']:.2f} GB)")
    print(f"  Avg input frames  : {mem_est['avg_frames_input']}")
    print(f"  Avg output T'     : {mem_est['avg_frames_output']}")

    n_train = split_data.get("train", {}).get("samples", 1)
    bs = args.batch_size
    steps_per_epoch = n_train / bs
    print(f"\n  Samples (train)   : {n_train}")
    print(f"  Steps per epoch   : ~{steps_per_epoch:.0f} "
          f"({n_train}/{bs} batches)")
    print(f"  Epochs (default)  : 100")
    print(f"  Total steps       : ~{steps_per_epoch * 100:.0f}")

    # Rough time estimate (assuming ~0.3-0.5s per step on a decent GPU)
    for sps in (0.2, 0.3, 0.5):
        epoch_time_min = steps_per_epoch * sps / 60
        total_time_h = steps_per_epoch * sps * 100 / 3600
        print(f"  @ {sps:.1f}s/step  : ~{epoch_time_min:.0f} min/epoch, "
              f"~{total_time_h:.0f}h total (100 epochs)")

    # ---- 8. Warnings & recommendations ----
    print(f"\n{' Warnings & Recommendations ':-^72}")

    # Check if real data is available
    train_data = split_data.get("train", {})
    if train_data.get("samples", 0) == 0:
        print(f"  ⚠ No train data found. Make sure --data_root points to")
        print(f"    the PHOENIX-2014T root with annotations/ and train/ folders.")
    else:
        # Check frame availability
        train_fs = frame_stats.get("train", {})
        if train_fs.get("missing_dirs", 0) > 0:
            print(f"  ⚠ {train_fs['missing_dirs']} training samples have no frame directory.")
            print(f"    Run with real extracted frames for proper frame stats.")

    # Known dataset quirks
    print(f"  ℹ  PHOENIX-2014T has 9 signers; train/dev/test contain different signers.")
    print(f"  ℹ  Glosses are German words in lowercase (e.g., 'ich', 'und', 'morgen').")
    print(f"  ℹ  Translations are full German sentences (not used for CSLR training).")
    print(f"  ℹ  CTC blank index = 0; '<unk>' = 1; valid glosses start at index 2.")

    print(f"\n{sep}\n")


def export_csv(split_data, frame_stats, gloss_stats, trans_stats, path: str):
    """Export per-sample details to CSV."""
    rows_out = []
    for split in ("train", "dev", "test"):
        data = split_data.get(split, {})
        csv_rows = data.get("rows", [])
        fs = frame_stats.get(split, {})
        gs = gloss_stats.get(split, {})

        frame_counts = fs.get("all_counts", [])
        gloss_lengths = gs.get("all_lengths", [])

        for i, row in enumerate(csv_rows):
            name = row.get("name", "").strip()
            n_frames = frame_counts[i] if i < len(frame_counts) else 0
            n_glosses = gloss_lengths[i] if i < len(gloss_lengths) else 0
            rows_out.append({
                "split": split,
                "name": name,
                "speaker": row.get("speaker", ""),
                "n_frames": n_frames,
                "n_glosses": n_glosses,
                "orth": row.get("orth", ""),
                "translation": row.get("translation", ""),
            })

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows_out[0].keys())
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"  Per-sample report exported to: {path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="PHOENIX-2014T Dataset Analysis"
    )
    parser.add_argument("--data_root", default="./data/phoenix",
                        help="Path to PHOENIX-2014T root directory")
    parser.add_argument("--output", default=None,
                        help="Export per-sample CSV report")
    parser.add_argument("--plots", action="store_true", default=True,
                        help="Generate plots (default: True)")
    parser.add_argument("--no-plots", action="store_false", dest="plots",
                        help="Skip plot generation")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for memory estimates")
    parser.add_argument("--d_model", type=int, default=512,
                        help="Model hidden dimension")
    parser.add_argument("--backbone", default="resnet18",
                        choices=["resnet18", "mobilenet_v3_small", "efficientnet_b0"])
    args = parser.parse_args()

    if not os.path.isdir(args.data_root):
        print(f"[!] Data root not found: {args.data_root}")
        print(f"    Point to the PHOENIX-2014T root containing "
              f"train/dev/test/ and annotation CSVs.")
        sys.exit(1)

    print(f"Data root: {args.data_root}")
    print(f"Scanning corpus CSVs and frame directories...\n")

    # ---- Parse all splits ----
    split_data = {}
    for split in ("train", "dev", "test"):
        csv_path = find_corpus_csv(args.data_root, split)
        if csv_path is None:
            print(f"  [{split}] No CSV found — skipping")
            split_data[split] = {"samples": 0, "csv_path": None, "rows": []}
            continue

        rows = load_csv(csv_path)
        print(f"  [{split}] Loaded {len(rows)} samples from CSV")

        # Speaker analysis
        speakers = analyze_speakers(rows)

        split_data[split] = {
            "csv_path": csv_path,
            "samples": len(rows),
            "rows": rows,
            "speakers": speakers,
        }

    # ---- Frame analysis ----
    print("\nAnalyzing frame counts...")
    frame_stats = {}
    for split in ("train", "dev", "test"):
        rows = split_data[split].get("rows", [])
        if not rows:
            frame_stats[split] = {}
            continue
        frame_stats[split] = analyze_frames(args.data_root, rows, split)
        fs = frame_stats[split]
        print(f"  [{split}] {fs.get('count', 0)} samples, "
              f"avg={fs.get('avg', 0):.0f} frames, "
              f"median={fs.get('p50', 0)}, "
              f"max={fs.get('max', 0)} "
              f"(missing_dirs={fs.get('missing_dirs', 0)})")

    # ---- Gloss analysis ----
    print("\nAnalyzing glosses...")
    gloss_stats = {}
    for split in ("train", "dev", "test"):
        rows = split_data[split].get("rows", [])
        gloss_stats[split] = analyze_glosses(rows)
        gs = gloss_stats[split]
        print(f"  [{split}] vocab={gs.get('vocab_size', 0)}, "
              f"avg_len={gs.get('avg_len', 0):.1f}, "
              f"max_len={gs.get('max_len', 0)}")

    # ---- Translation analysis ----
    print("\nAnalyzing translations...")
    trans_stats = {}
    for split in ("train", "dev", "test"):
        rows = split_data[split].get("rows", [])
        trans_stats[split] = analyze_translations(rows)
        ts = trans_stats[split]
        print(f"  [{split}] {ts.get('count', 0)} sentences, "
              f"avg={ts.get('avg_len', 0):.1f} words, "
              f"vocab={ts.get('vocab_size', 0)} unique words")

    # ---- Memory estimate ----
    train_fs = frame_stats.get("train", {})
    avg_frames = train_fs.get("avg", PHOENIX_KNOWN["avg_frames_raw"])
    mem_est = estimate_memory(
        avg_frames=avg_frames,
        batch_size=args.batch_size,
        d_model=args.d_model,
        backbone=args.backbone,
        vocab_size=gloss_stats.get("train", {}).get("vocab_size", 1088),
    )

    # ---- Print report ----
    print_report(split_data, frame_stats, gloss_stats, trans_stats, mem_est, args)

    # ---- Plots ----
    if args.plots:
        print("Generating plots...")
        make_plots(split_data, frame_stats, gloss_stats, trans_stats)

    # ---- CSV export ----
    if args.output:
        export_csv(split_data, frame_stats, gloss_stats, trans_stats, args.output)

    print("Done.")


if __name__ == "__main__":
    main()
