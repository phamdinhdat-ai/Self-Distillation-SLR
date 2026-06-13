"""
Entry point for SMKD training. Run with `python train.py [options]`.

Examples:
    # Synthetic data smoke test (no dataset needed, tiny model)
    python train.py --synthetic --epochs 6 --d_model 32 --hidden_size 32 \\
        --img_size 32 --max_frames 16 --stage1_end 2 --stage2_end 4 --eval_every 1

    # Real PHOENIX14-T subset, single/auto GPU
    python train.py --real --data_root /path/to/phoenix2014T

    # Real data, 2 GPUs with DataParallel (default when multi_gpu not disabled)
    python train.py --real --data_root /path/to/phoenix2014T --batch_size 4

    # Real data, force single-GPU/CPU even if multiple GPUs are visible
    python train.py --real --data_root /path/to/phoenix2014T --no-multi-gpu

    # After training, save a WER/loss plot to disk
    python train.py --real --data_root /path/to/phoenix2014T --plot_path ./wer_history.png
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trainer import Trainer


def parse_args():
    p = argparse.ArgumentParser(
        description="SMKD CSLR Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- Data ----
    data_group = p.add_argument_group("data")
    data_group.add_argument(
        "--data_root", default="./data/phoenix",
        help="Path to PHOENIX14-T root (contains <ann_dir>/, train/, dev/, test/)",
    )
    # synthetic vs real toggle: default is synthetic=True (no dataset needed)
    data_group.add_argument(
        "--synthetic", dest="synthetic", action="store_true",
        help="Use synthetic random data (default)",
    )
    data_group.add_argument(
        "--real", "--no-synthetic", dest="synthetic", action="store_false",
        help="Use real data from --data_root",
    )
    p.set_defaults(synthetic=True)

    data_group.add_argument("--img_size", type=int, default=224,
                             help="Frame crop size (HxW)")
    data_group.add_argument("--max_frames", type=int, default=300,
                             help="Max frames per sequence (longer sequences are subsampled)")
    data_group.add_argument("--num_workers", type=int, default=0,
                             help="DataLoader worker processes")

    # ---- Model ----
    model_group = p.add_argument_group("model")
    model_group.add_argument("--d_model", type=int, default=512,
                              help="Feature dimension for visual/contextual modules")
    model_group.add_argument("--hidden_size", type=int, default=512,
                              help="BiLSTM hidden size (per direction)")

    # ---- Optimization ----
    opt_group = p.add_argument_group("optimization")
    opt_group.add_argument("--batch_size", type=int, default=2)
    opt_group.add_argument("--lr", type=float, default=1e-4)
    opt_group.add_argument("--lr_milestones", type=int, nargs="+", default=[40, 60, 80],
                            help="Epochs at which LR is halved (stages 1 & 2 only)")
    opt_group.add_argument("--alpha", type=float, default=0.5,
                            help="Weight balancing visual vs contextual/seg loss")
    opt_group.add_argument("--label_smoothing", type=float, default=0.2,
                            help="Label smoothing for the GSBA segmentation CE loss")

    # ---- Training schedule ----
    sched_group = p.add_argument_group("schedule")
    sched_group.add_argument("--epochs", type=int, default=100)
    sched_group.add_argument("--stage1_end", type=int, default=30,
                              help="Last epoch of stage 1 (synchronous training)")
    sched_group.add_argument("--stage2_end", type=int, default=90,
                              help="Last epoch of stage 2 (gloss segmentation)")
    sched_group.add_argument("--gsba_update_every", type=int, default=10,
                              help="GSBA radius d grows by 1 every N epochs (during stage 2)")
    sched_group.add_argument("--eval_every", type=int, default=5,
                              help="Run dev WER evaluation every N epochs")

    # ---- Hardware ----
    hw_group = p.add_argument_group("hardware")
    hw_group.add_argument("--multi_gpu", dest="multi_gpu", action="store_true",
                           help="Use nn.DataParallel across all visible GPUs (default)")
    hw_group.add_argument("--no-multi-gpu", dest="multi_gpu", action="store_false",
                           help="Force single-GPU/CPU even if multiple GPUs are visible")
    p.set_defaults(multi_gpu=True)

    # ---- Output ----
    out_group = p.add_argument_group("output")
    out_group.add_argument("--ckpt_dir", default="./checkpoints",
                            help="Directory for best-checkpoint .pt files")
    out_group.add_argument("--plot_path", default=None,
                            help="If set, save a WER/loss history plot to this path after training")
    out_group.add_argument("--print_history", action="store_true",
                            help="Print a text table of the per-epoch history after training")

    return p.parse_args()


def main():
    args = parse_args()
    config = vars(args)

    if config["synthetic"]:
        print("Using SYNTHETIC data (no dataset needed)")
    else:
        print(f"Using REAL data from: {config['data_root']}")
        if not os.path.isdir(config["data_root"]):
            raise FileNotFoundError(
                f"--data_root '{config['data_root']}' does not exist. "
                f"Pass --data_root /path/to/phoenix2014T (containing "
                f"<ann_dir>/PHOENIX-2014-T.*.corpus.csv and train/dev/test/<name>/ folders)."
            )

    trainer = Trainer(config)
    trainer.train()

    if args.print_history:
        trainer.print_history_table()

    if args.plot_path:
        trainer.plot_history(save_path=args.plot_path, show=False)


if __name__ == "__main__":
    main()