"""
Entry point for SMKD training.

Usage:
    python train.py                       # synthetic data (unit test)
    python train.py --data_root /path/to/phoenix14
"""

import argparse
import sys
import os

# Make project root importable
sys.path.insert(0, os.path.dirname(__file__))

from trainer import Trainer


def parse_args():
    p = argparse.ArgumentParser(description="SMKD CSLR Training")
    p.add_argument("--data_root",   default="./data/phoenix", help="Path to PHOENIX14 root")
    p.add_argument("--synthetic",   action="store_true", default=True,
                   help="Use synthetic data (no real dataset needed)")
    p.add_argument("--epochs",      type=int,   default=100)
    p.add_argument("--batch_size",  type=int,   default=2)
    p.add_argument("--d_model",     type=int,   default=512)
    p.add_argument("--hidden_size", type=int,   default=512)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--alpha",       type=float, default=0.5,
                   help="Weight balancing visual vs contextual CTC loss")
    p.add_argument("--label_smoothing", type=float, default=0.2)
    p.add_argument("--stage1_end",  type=int,   default=30)
    p.add_argument("--stage2_end",  type=int,   default=90)
    p.add_argument("--ckpt_dir",    default="./checkpoints")
    p.add_argument("--num_workers", type=int,   default=0)
    return p.parse_args()


def main():
    args   = parse_args()
    config = vars(args)
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
