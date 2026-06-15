#!/usr/bin/env python3
"""
Pre-process PHOENIX14-T frames and cache them as .pt tensors.

This runs through all samples once, decodes images, applies Resize+ToTensor,
and saves (T, 3, 256, 256) tensors to a cache directory. Subsequent training
runs with cache_dir set will load from cache instead of re-processing images.

Usage:
    # Pre-cache training set
    python experiments/preprocess.py --data_root /path/to/phoenix2014T --split train

    # Pre-cache all splits
    python experiments/preprocess.py --data_root /path/to/phoenix2014T --split train dev test

    # Specify cache directory and image size
    python experiments/preprocess.py --data_root /path/to/phoenix2014T --split train \\
        --cache_dir ./data/cache --img_size 224 --max_frames 300
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from dataset import PhoenixDataset, build_cache_transform


def main():
    p = argparse.ArgumentParser(description="Pre-cache PHOENIX14-T frames as .pt tensors")
    p.add_argument("--data_root", required=True, help="Path to PHOENIX14-T root")
    p.add_argument("--split", nargs="+", default=["train"],
                   help="Splits to pre-cache (train dev test)")
    p.add_argument("--cache_dir", default="./data/cache", help="Cache output directory")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--max_frames", type=int, default=300)
    args = p.parse_args()

    cache_transform = build_cache_transform()

    for split in args.split:
        print(f"\n{'='*60}")
        print(f"  Pre-caching split: {split}")
        print(f"{'='*60}")

        ds = PhoenixDataset(
            root=args.data_root,
            split=split,
            img_size=args.img_size,
            max_frames=args.max_frames,
            synthetic=False,
        )

        os.makedirs(args.cache_dir, exist_ok=True)
        total_frames = 0
        total_size_mb = 0.0
        t0 = time.time()

        for i, sample in enumerate(ds.samples):
            sample_id = sample["id"]
            cache_path = os.path.join(
                args.cache_dir,
                f"{sample_id}_sz{args.img_size}_mf{args.max_frames}.pt"
            )

            if os.path.exists(cache_path):
                print(f"  [{i+1:4d}/{len(ds)}] {sample_id:<45s}  (cached)")
                continue

            try:
                frames = ds._load_frames_raw(sample["frame_dir"])
                torch.save(frames, cache_path)
                n_frames = frames.size(0)
                size_mb = frames.numel() * 4 / (1024 * 1024)
                total_frames += n_frames
                total_size_mb += size_mb
                print(f"  [{i+1:4d}/{len(ds)}] {sample_id:<45s}  "
                      f"{n_frames:3d}f  {size_mb:5.1f}MB")
            except Exception as e:
                print(f"  [{i+1:4d}/{len(ds)}] {sample_id:<45s}  ERROR: {e}")

        elapsed = time.time() - t0
        print(f"\n  Split '{split}' done: {len(ds)} samples, "
              f"{total_frames} frames, {total_size_mb:.0f} MB cached")
        print(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f}m)")

    print(f"\n  Cache directory: {args.cache_dir}")
    print(f"  To use: set data.cache_dir: \"{args.cache_dir}\" in your YAML config")


if __name__ == "__main__":
    main()
