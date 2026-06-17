#!/usr/bin/env python3
"""
Data loading latency benchmark for SMKD training.

Measures where time is spent in the training loop by instrumenting each phase:
  1. DataLoader batch fetch (disk I/O + image decode + transforms)
  2. Host → GPU transfer (.to(device))
  3. Forward pass (backbone + temporal CNN + BiLSTM + classifier)
  4. Loss computation

Also reports throughput (samples/sec, batches/sec) and GPU utilization hints.

Usage:
    python experiments/benchmark_loader.py
    python experiments/benchmark_loader.py --num_workers 0 4 8
    python experiments/benchmark_loader.py --batch_size 4 8 16
    python experiments/benchmark_loader.py --backbone resnet18
    python experiments/benchmark_loader.py --no-model  # pure data loading only
"""

import argparse
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import PhoenixDataset, collate_fn
from models.smkd import SMKD


# ============================================================================
# Helpers
# ============================================================================

def fmt_ms(seconds: float) -> str:
    """Format seconds as ms or s depending on magnitude."""
    if seconds < 0.001:
        return f"{seconds * 1_000_000:.1f} µs"
    elif seconds < 1.0:
        return f"{seconds * 1_000:.1f} ms"
    elif seconds < 60:
        return f"{seconds:.2f} s"
    else:
        m, s = divmod(int(seconds), 60)
        return f"{m}m{s:02d}s"


def fmt_rate(count: int, seconds: float) -> str:
    if seconds <= 0:
        return "∞"
    rate = count / seconds
    if rate < 1:
        return f"{rate:.2f}/s"
    return f"{rate:.1f}/s"


def pct_str(part: float, total: float) -> str:
    if total <= 0:
        return "  - %"
    return f"{part / total * 100:5.1f}%"


# ============================================================================
# Benchmark
# ============================================================================

def benchmark(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpus = torch.cuda.device_count() if device.type == "cuda" else 0

    # ---- Header ----
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  SMKD Data Loading Benchmark")
    print(f"  Device: {device} ({n_gpus} GPUs)" if n_gpus else f"  Device: {device}")
    print(f"  Data root: {args.data_root}")
    print(sep)

    # ---- Build dataset ----
    print(f"\n  Loading dataset...")
    t0 = time.time()
    train_ds = PhoenixDataset(
        root=args.data_root,
        split="train",
        img_size=args.img_size,
        max_frames=args.max_frames,
        synthetic=args.synthetic,
    )
    n_samples = len(train_ds)
    vocab_size = len(train_ds.vocab)
    load_time = time.time() - t0
    print(f"  Dataset: {n_samples} samples, vocab={vocab_size}, "
          f"loaded in {fmt_ms(load_time)}")

    # ---- Build model (optional) ----
    model = None
    if not args.no_model:
        print(f"  Building SMKD model (backbone={args.backbone}, "
              f"d_model={args.d_model})...")
        model = SMKD(
            num_classes=vocab_size,
            d_model=args.d_model,
            hidden_size=args.hidden_size,
            backbone=args.backbone,
            temporal_conv_type=args.temporal_conv_type,
            pretrained_backbone=args.pretrained_backbone,
            freeze_backbone=args.freeze_backbone,
        ).to(device)
        model.set_stage(1)
        model.eval()
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Model: {total_params:,} params ({total_params * 4 / 1e6:.1f} MB FP32)")

    # ---- Run benchmark for each config ----
    for nw in args.num_workers_list:
        for bs in args.batch_size_list:
            print(f"\n{'—' * 72}")
            print(f"  num_workers={nw}, batch_size={bs}, "
                  f"num_batches={args.num_batches}")

            loader = torch.utils.data.DataLoader(
                train_ds, batch_size=bs, shuffle=True,
                collate_fn=collate_fn, num_workers=nw,
                pin_memory=(device.type == "cuda"),
                drop_last=True,
            )

            timings = run_benchmark(loader, model, device, args)
            print_results(timings, bs, nw, n_samples, device)

            if args.num_batches <= 10 and args.detailed:
                print_detailed_timeline(timings)


def run_benchmark(
    loader, model, device, args
) -> Dict[str, List[float]]:
    """
    Run `num_batches` through the loader + optional model, recording
    microsecond-precision timings at each phase.

    Timing strategy: measure the producer-consumer pipeline.
    - "loader_wait": wall-clock time spent waiting for the next batch
                     (GPU idle while CPU loads/decodes images).
    - "h2d": Host→Device copy time.
    - "forward": model forward pass (synchronised via cuda.synchronize).
    - "loss": log-softmax + placeholder loss (synchronised).
    - "total": end-to-end time per step (= loader_wait + h2d + forward + loss).
    """
    # Warm-up: 3 batches to fill CUDA allocator / worker queues
    warmup = min(3, args.num_batches)
    for i, batch in enumerate(loader):
        if i >= warmup:
            break
        if model is not None:
            frames = batch["frames"].to(device, non_blocking=True)
            with torch.no_grad(), torch.cuda.amp.autocast() if args.use_amp else torch.enable_grad():
                _ = model(frames)

    # Timed runs — measure the pipeline properly
    timings = defaultdict(list)

    # Prime the iterator: fetch first batch, measure how long it takes
    loader_iter = iter(loader)
    t0 = time.perf_counter()
    try:
        batch = next(loader_iter)
    except StopIteration:
        return timings
    t_got_batch = time.perf_counter()
    # First batch loader_wait = initial fetch time
    loader_wait_0 = t_got_batch - t0

    batch_count = 0
    while batch_count < args.num_batches:
        # ---- Process this batch (model) ----
        if model is not None:
            # H→D
            frames = batch["frames"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            t_h2d_done = time.perf_counter()
            timings["h2d"].append(t_h2d_done - t_got_batch)

            # Forward
            with torch.no_grad():
                if args.use_amp:
                    with torch.cuda.amp.autocast():
                        out = model(frames)
                else:
                    out = model(frames)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_fwd_done = time.perf_counter()
            timings["forward"].append(t_fwd_done - t_h2d_done)

            # Loss
            log_probs = F.log_softmax(out["logits_g"].float(), dim=-1)
            _ = log_probs.sum()
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_model_done = time.perf_counter()
            timings["loss"].append(t_model_done - t_fwd_done)
        else:
            t_model_done = t_got_batch

        # ---- Start fetching NEXT batch while recording loader wait ----
        t_start_fetch = time.perf_counter()
        try:
            batch = next(loader_iter)
        except StopIteration:
            break
        t_got_next = time.perf_counter()

        # loader_wait = how long GPU was idle waiting for this batch
        loader_wait = t_got_next - t_start_fetch
        if batch_count == 0:
            loader_wait = loader_wait_0  # first batch was measured above

        timings["loader_wait"].append(loader_wait)
        timings["total"].append(t_got_next - t_got_batch)
        batch_count += 1

        t_got_batch = t_got_next

    return timings


def print_results(
    timings: Dict[str, List[float]],
    batch_size: int,
    num_workers: int,
    n_samples: int,
    device,
):
    """Print a formatted results table for one (num_workers, batch_size) combo."""
    n = len(timings["total"])
    if n == 0:
        print("  (no batches completed)")
        return

    total_s = sum(timings["total"])

    def avg_ms(key):
        vals = timings.get(key, [])
        return sum(vals) / len(vals) * 1000 if vals else 0

    loader_ms = avg_ms("loader_wait")
    h2d_ms = avg_ms("h2d")
    forward_ms = avg_ms("forward")
    loss_ms = avg_ms("loss")
    total_ms = avg_ms("total")

    overhead_ms = total_ms - (loader_ms + h2d_ms + forward_ms + loss_ms)

    # Throughput
    samples_per_sec = (n * batch_size) / total_s if total_s > 0 else 0
    batches_per_sec = n / total_s if total_s > 0 else 0
    epochs_per_hour = samples_per_sec * 3600 / n_samples if n_samples > 0 else 0

    print(f"\n  {'Phase':<20s} {'Avg/step':>10s} {'%':>6s}  {'Total':>10s}")
    print(f"  {'-'*20} {'-'*10} {'-'*6}  {'-'*10}")
    print(f"  {'GPU idle (loader)':<20s} {loader_ms:>8.1f} ms "
          f"{pct_str(loader_ms, total_ms):>6s}  {fmt_ms(sum(timings['loader_wait'])):>10s}")
    if "h2d" in timings and timings["h2d"]:
        print(f"  {'Host→Device copy':<20s} {h2d_ms:>8.1f} ms "
              f"{pct_str(h2d_ms, total_ms):>6s}  {fmt_ms(sum(timings['h2d'])):>10s}")
    if "forward" in timings and timings["forward"]:
        print(f"  {'Forward pass':<20s} {forward_ms:>8.1f} ms "
              f"{pct_str(forward_ms, total_ms):>6s}  {fmt_ms(sum(timings['forward'])):>10s}")
    if "loss" in timings and timings["loss"]:
        print(f"  {'Loss compute':<20s} {loss_ms:>8.1f} ms "
              f"{pct_str(loss_ms, total_ms):>6s}  {fmt_ms(sum(timings['loss'])):>10s}")
    if abs(overhead_ms) > 0.01:
        print(f"  {'Overhead/other':<20s} {overhead_ms:>8.1f} ms "
              f"{pct_str(overhead_ms, total_ms):>6s}")
    print(f"  {'─'*20} {'─'*10} {'─'*6}  {'─'*10}")
    print(f"  {'TOTAL per step':<20s} {total_ms:>8.1f} ms            "
          f"{fmt_ms(total_s):>10s}")

    print(f"\n  Throughput:")
    print(f"    {batches_per_sec:.1f} batches/s  |  "
          f"{samples_per_sec:.1f} samples/s  |  "
          f"{epochs_per_hour:.2f} epochs/h")
    print(f"    Estimated 1 epoch: {fmt_ms(n_samples / samples_per_sec if samples_per_sec > 0 else 0)}")
    print(f"    Estimated 100 epochs: {fmt_ms(100 * n_samples / samples_per_sec if samples_per_sec > 0 else 0)}")

    # Bottleneck diagnosis
    print(f"\n  Bottleneck analysis:")
    gpu_busy_ms = forward_ms + h2d_ms + loss_ms
    if gpu_busy_ms > 0:
        idle_pct = loader_ms / total_ms * 100 if total_ms > 0 else 0
        gpu_util = max(0, 100 - idle_pct)
        print(f"    GPU busy time    : {gpu_busy_ms:.0f} ms/step")
        print(f"    GPU idle (loader): {loader_ms:.0f} ms/step")
        print(f"    Est. GPU util    : ~{gpu_util:.0f}%")

    no_model = (forward_ms + h2d_ms + loss_ms) < 0.01
    if not no_model and loader_ms > max(gpu_busy_ms, 0.001) * 1.5:
        print(f"    ⚠  DATA-LOADING BOUND — GPU idle {loader_ms:.0f}ms vs busy "
              f"{gpu_busy_ms:.0f}ms")
        print(f"       → Increase num_workers (currently {num_workers})")
        print(f"       → Consider pre-caching images as .npy/.pt files")
        print(f"       → Use DALI or turbojpeg for faster decode")
    elif not no_model and forward_ms > loader_ms * 1.5:
        print(f"    ⚠  COMPUTE-BOUND — Forward pass takes "
              f"{forward_ms:.0f}ms vs DataLoader {loader_ms:.0f}ms")
        print(f"       → Enable AMP (use_amp=True)")
        print(f"       → Use a lighter backbone")
    elif not no_model:
        ratio = loader_ms / max(gpu_busy_ms, 0.001)
        print(f"    ✓  BALANCED — Idle/Busy ratio = {ratio:.1f}x")
        print(f"       Loader wait: {loader_ms:.0f}ms, GPU busy: {gpu_busy_ms:.0f}ms")


def print_detailed_timeline(timings: Dict[str, List[float]]):
    """Print per-batch breakdown for first batches."""
    n = min(len(timings["total"]), 5)
    print(f"\n  Per-batch detail (first {n} batches):")
    header = f"  {'Batch':>6s}"
    cols = []
    for key in ["loader_wait", "h2d", "forward", "loss", "total"]:
        if key in timings and timings[key]:
            header += f"  {' ' + key:>12s}"
            cols.append(key)
    if not cols:
        return
    print(header)
    for i in range(n):
        row = f"  {i:>6d}"
        for key in cols:
            ms = timings[key][i] * 1000 if i < len(timings[key]) else 0
            row += f"  {ms:>10.1f} ms"
        print(row)
    print()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="SMKD Data Loading Latency Benchmark"
    )
    parser.add_argument("--data_root", default="./data/phoenix",
                        help="Path to PHOENIX-2014T root")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data (skip real frames)")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--max_frames", type=int, default=300)

    # Model options
    parser.add_argument("--no-model", action="store_true",
                        help="Measure data loading only (no model)")
    parser.add_argument("--backbone", default="resnet18",
                        choices=["resnet18", "mobilenet_v3_small", "efficientnet_b0"])
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--temporal_conv_type", default="standard")
    parser.add_argument("--pretrained_backbone", action="store_true")
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--use_amp", action="store_true",
                        help="Enable AMP mixed precision for forward pass")

    # Sweep options
    parser.add_argument("--num_workers", type=int, nargs="*", default=[0],
                        help="List of num_workers values to test (default: 0)")
    parser.add_argument("--batch_size", type=int, nargs="*", default=[4],
                        help="List of batch_size values to test (default: 4)")
    parser.add_argument("--num_batches", type=int, default=20,
                        help="Number of batches to benchmark (default: 20)")

    parser.add_argument("--detailed", action="store_true", default=True,
                        help="Show per-batch timeline (default: True)")
    parser.add_argument("--no-detailed", action="store_false", dest="detailed",
                        help="Hide per-batch timeline")

    args = parser.parse_args()
    args.num_workers_list = args.num_workers
    args.batch_size_list = args.batch_size

    if args.synthetic:
        print("  [synthetic mode — random tensors, no disk I/O]")

    benchmark(args)


if __name__ == "__main__":
    main()
