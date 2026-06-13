"""
Three-stage Trainer for SMKD, with pretty console output, history tracking,
and optional multi-GPU training via nn.DataParallel.

Stage schedule (default):
  epochs  1..stage1_end             -> Stage 1: synchronous training (shared classifier)
  epochs  stage1_end+1..stage2_end  -> Stage 2: gloss segmentation (GSBA)
  epochs  stage2_end+1..epochs      -> Stage 3: decouple training

After training, `trainer.history` contains per-epoch loss / WER values that
can be plotted with `trainer.plot_history()`.

Multi-GPU:
  Set cfg["multi_gpu"] = True (default) to automatically wrap the model in
  nn.DataParallel when more than one CUDA device is visible. With 2 GPUs,
  each forward/backward splits the batch across both devices.

  Practical notes for 2-GPU DataParallel:
    - Effective batch size is split across GPUs, so batch_size=2 means
      1 sample per GPU. For CTC loss (which needs >=1 sample per device to
      avoid empty shards), prefer batch_size >= 2 * num_gpus.
    - nn.CTCLoss expects (T, B, C) and is computed inside SMKDLoss on the
      gathered outputs (after DataParallel re-combines per-GPU outputs on
      the default device), so the loss itself is NOT replicated - this is
      handled correctly here.
    - model.set_stage(...) and direct attribute access (e.g.
      model.shared_classifier.weight) must go through `self.raw_model`,
      which always points at the underlying (un-wrapped) SMKD module.
"""

import os
import time
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

from models.smkd import SMKD
from losses import SMKDLoss, batch_gsba
from dataset import PhoenixDataset, collate_fn
from metrics import compute_wer


# ---------------------------------------------------------------------------
# Decoding helper
# ---------------------------------------------------------------------------

def greedy_ctc_decode(logits: torch.Tensor, blank: int = 0) -> List[List[int]]:
    """logits: (B, T', C+1) -> list of B decoded sequences (no blanks/repeats)."""
    preds = logits.argmax(dim=-1).cpu().tolist()
    results = []
    for seq in preds:
        decoded, prev = [], blank
        for tok in seq:
            if tok != blank and tok != prev:
                decoded.append(tok)
            prev = tok
        results.append(decoded)
    return results


# ---------------------------------------------------------------------------
# Pretty-printing helpers
# ---------------------------------------------------------------------------

_BOX_WIDTH = 70


def _hr(char: str = "-") -> str:
    return char * _BOX_WIDTH


def _box_title(title: str) -> str:
    pad = max(0, _BOX_WIDTH - len(title) - 2)
    left = pad // 2
    right = pad - left
    return f"{'=' * left} {title} {'=' * right}"


def _fmt_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:4.1f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m:2d}m{s:02d}s"


STAGE_NAMES = {
    1: "Synchronous training (shared classifier)",
    2: "Gloss segmentation (GSBA)",
    3: "Decouple training",
}


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ---- Multi-GPU setup ----
        self.n_gpus = torch.cuda.device_count()
        self.use_data_parallel = (
            cfg.get("multi_gpu", True)
            and self.device.type == "cuda"
            and self.n_gpus > 1
        )

        print(_box_title("Loading datasets"))
        train_ds = PhoenixDataset(
            root=cfg.get("data_root", "./data/phoenix"),
            split="train",
            img_size=cfg.get("img_size", 224),
            max_frames=cfg.get("max_frames", 300),
            synthetic=cfg.get("synthetic", True),
        )
        val_ds = PhoenixDataset(
            root=cfg.get("data_root", "./data/phoenix"),
            split="dev",
            vocab=train_ds.vocab,
            img_size=cfg.get("img_size", 224),
            max_frames=cfg.get("max_frames", 300),
            synthetic=cfg.get("synthetic", True),
        )
        self.vocab = train_ds.vocab

        batch_size = cfg.get("batch_size", 2)
        if self.use_data_parallel and batch_size < self.n_gpus:
            print(f"  [!] batch_size={batch_size} < n_gpus={self.n_gpus}; "
                  f"some GPUs would get empty shards. Consider batch_size "
                  f">= {self.n_gpus} (ideally a multiple of {self.n_gpus}).")

        self.train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            collate_fn=collate_fn, num_workers=cfg.get("num_workers", 0),
            pin_memory=(self.device.type == "cuda"),
            drop_last=self.use_data_parallel,  # avoid uneven last-batch GPU splits
        )
        self.val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=cfg.get("num_workers", 0),
            pin_memory=(self.device.type == "cuda"),
        )

        # ---- Model ----
        num_classes = len(self.vocab)
        raw_model = SMKD(
            num_classes=num_classes,
            d_model=cfg.get("d_model", 512),
            hidden_size=cfg.get("hidden_size", 512),
        ).to(self.device)

        if self.use_data_parallel:
            self.model = nn.DataParallel(raw_model)
        else:
            self.model = raw_model

        # `raw_model` always points at the underlying SMKD module, even when
        # wrapped in DataParallel. Use this for set_stage(), direct param
        # access (e.g. shared_classifier.weight), and checkpointing.
        self.raw_model = raw_model

        self.criterion = SMKDLoss(
            blank=0,
            alpha=cfg.get("alpha", 0.5),
            label_smoothing=cfg.get("label_smoothing", 0.2),
        )

        self.optimizer = Adam(self.model.parameters(), lr=cfg.get("lr", 1e-4))
        self.scheduler = MultiStepLR(
            self.optimizer, milestones=cfg.get("lr_milestones", [40, 60, 80]), gamma=0.5,
        )

        self.stage1_end = cfg.get("stage1_end", 30)
        self.stage2_end = cfg.get("stage2_end", 90)
        self.total_epochs = cfg.get("epochs", 100)
        self.eval_every = cfg.get("eval_every", 5)

        self.gsba_d = 1
        self.gsba_update_every = cfg.get("gsba_update_every", 10)

        self.best_wer = float("inf")
        self.best_epoch = None
        self.ckpt_dir = cfg.get("ckpt_dir", "./checkpoints")
        os.makedirs(self.ckpt_dir, exist_ok=True)

        # Per-epoch history for plotting / inspection.
        # `wer` entries are None for epochs where evaluation didn't run.
        self.history = {
            "epoch": [],
            "stage": [],
            "lr": [],
            "ctc_g": [],
            "ctc_v": [],
            "seg": [],
            "total": [],
            "wer": [],
            "time": [],
        }

        self._print_setup_summary()

    # ------------------------------------------------------------------
    # Setup summary
    # ------------------------------------------------------------------

    def _print_setup_summary(self):
        n_params = sum(p.numel() for p in self.raw_model.parameters())
        print(_box_title("SMKD Trainer"))
        print(f"  device          : {self.device}")
        if self.device.type == "cuda":
            gpu_names = ", ".join(
                torch.cuda.get_device_name(i) for i in range(self.n_gpus)
            )
            print(f"  GPUs visible    : {self.n_gpus} ({gpu_names})")
            print(f"  DataParallel    : {'ENABLED' if self.use_data_parallel else 'disabled'}")
        print(f"  vocab size      : {len(self.vocab)}")
        print(f"  train / dev     : {len(self.train_loader.dataset)} / "
              f"{len(self.val_loader.dataset)} samples")
        print(f"  batch size      : {self.cfg.get('batch_size', 2)}"
              + (f"  ({self.cfg.get('batch_size', 2) // max(self.n_gpus,1)} per GPU)"
                 if self.use_data_parallel else ""))
        print(f"  model dims      : d_model={self.raw_model.d_model}, "
              f"params={n_params:,}")
        print(f"  total epochs    : {self.total_epochs}")
        print(f"  stage schedule  :")
        s1_end = min(self.stage1_end, self.total_epochs)
        print(f"      stage 1 (epochs   1-{s1_end:>3}) : {STAGE_NAMES[1]}")
        if self.stage1_end < self.total_epochs:
            s2_end = min(self.stage2_end, self.total_epochs)
            print(f"      stage 2 (epochs {self.stage1_end+1:>3}-{s2_end:>3}) : {STAGE_NAMES[2]}")
        else:
            print(f"      stage 2 : (not reached - total_epochs <= stage1_end)")
        if self.stage2_end < self.total_epochs:
            print(f"      stage 3 (epochs {self.stage2_end+1:>3}-{self.total_epochs:>3}) : {STAGE_NAMES[3]}")
        else:
            print(f"      stage 3 : (not reached - total_epochs <= stage2_end)")
        print(f"  eval every      : {self.eval_every} epochs")
        print(f"  checkpoint dir  : {self.ckpt_dir}")
        print(_hr("="))

    # ------------------------------------------------------------------
    # Stage helpers
    # ------------------------------------------------------------------

    def _current_stage(self, epoch: int) -> int:
        if epoch <= self.stage1_end:
            return 1
        elif epoch <= self.stage2_end:
            return 2
        return 3

    def _maybe_update_gsba_d(self, epoch: int):
        if epoch > self.stage1_end:
            self.gsba_d = 1 + (epoch - self.stage1_end - 1) // 20

    def _current_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    def _set_stage(self, stage: int):
        """Sets the training stage on the underlying (unwrapped) model."""
        self.raw_model.set_stage(stage)

    # ------------------------------------------------------------------
    # Train step
    # ------------------------------------------------------------------

    def _train_step(self, batch: dict, stage: int):
        frames = batch["frames"].to(self.device, non_blocking=True)
        labels = batch["labels"].to(self.device, non_blocking=True)
        input_lengths = batch["input_lengths"].to(self.device, non_blocking=True)
        target_lengths = batch["target_lengths"].to(self.device, non_blocking=True)
        gloss_seqs = batch["gloss_seqs"]

        # With DataParallel, `self.model(frames)` runs the forward pass on
        # each GPU shard and gathers outputs back onto self.device.
        out = self.model(frames)
        logits_v, logits_g, gcf = out["logits_v"], out["logits_g"], out["gcf"]

        T_prime = logits_v.size(1)
        adj_lengths = torch.clamp(input_lengths // 2, max=T_prime)

        seg_labels = None
        if stage == 2:
            with torch.no_grad():
                spike_labels = logits_g.argmax(dim=-1)
                # Always read the shared classifier weight from raw_model -
                # under DataParallel, self.model.shared_classifier would not
                # exist (it's `self.model.module.shared_classifier`).
                w_n = F.normalize(self.raw_model.shared_classifier.weight, p=2, dim=-1)
                seg_labels = batch_gsba(
                    gcf=gcf.detach(), classifier_weights=w_n.detach(),
                    spike_labels=spike_labels, gloss_seqs=gloss_seqs, d=self.gsba_d,
                )

        loss, loss_dict = self.criterion(
            logits_v=logits_v, logits_g=logits_g, targets=labels,
            input_lengths=adj_lengths, target_lengths=target_lengths,
            stage=stage, seg_labels=seg_labels,
        )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
        self.optimizer.step()
        return loss_dict

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self) -> float:
        self.model.eval()
        all_hyps, all_refs = [], []

        for batch in self.val_loader:
            frames = batch["frames"].to(self.device, non_blocking=True)
            labels = batch["labels"]
            target_lengths = batch["target_lengths"]

            out = self.model(frames)
            hyps = greedy_ctc_decode(out["logits_g"])

            offset = 0
            for i, tlen in enumerate(target_lengths.tolist()):
                ref = labels[offset: offset + tlen].tolist()
                all_refs.append(ref)
                all_hyps.append(hyps[i])
                offset += tlen

        wer = compute_wer(all_hyps, all_refs)
        self.model.train()
        return wer

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self):
        print(_box_title("Training started"))
        train_start = time.time()

        for epoch in range(1, self.total_epochs + 1):
            epoch_start = time.time()

            stage = self._current_stage(epoch)
            self._maybe_update_gsba_d(epoch)

            if self.raw_model.stage != stage:
                self._set_stage(stage)
                print()
                print(_box_title(f"Stage {stage}: {STAGE_NAMES[stage]}"))
                if stage == 3:
                    self.optimizer = Adam(self.model.parameters(), lr=4e-6)

            # --- run one epoch ---
            epoch_losses = {}
            for batch in self.train_loader:
                loss_dict = self._train_step(batch, stage)
                for k, v in loss_dict.items():
                    epoch_losses[k] = epoch_losses.get(k, 0.0) + v

            n_batches = max(len(self.train_loader), 1)
            avg_losses = {k: v / n_batches for k, v in epoch_losses.items()}

            # --- evaluation ---
            run_eval = (epoch % self.eval_every == 0) or (epoch == self.total_epochs)
            wer = None
            improved = False
            if run_eval:
                wer = self.evaluate()
                if wer < self.best_wer:
                    self.best_wer = wer
                    self.best_epoch = epoch
                    improved = True
                    self._save_checkpoint(epoch, wer)

            epoch_time = time.time() - epoch_start

            # --- record history ---
            self.history["epoch"].append(epoch)
            self.history["stage"].append(stage)
            self.history["lr"].append(self._current_lr())
            self.history["ctc_g"].append(avg_losses.get("ctc_g"))
            self.history["ctc_v"].append(avg_losses.get("ctc_v"))
            self.history["seg"].append(avg_losses.get("seg"))
            self.history["total"].append(avg_losses.get("total"))
            self.history["wer"].append(wer)
            self.history["time"].append(epoch_time)

            # --- pretty print ---
            self._print_epoch_line(epoch, stage, avg_losses, wer, improved, epoch_time)

            if stage < 3:
                self.scheduler.step()

        total_time = time.time() - train_start
        self._print_final_summary(total_time)

    # ------------------------------------------------------------------
    # Pretty printing
    # ------------------------------------------------------------------

    def _print_epoch_line(self, epoch, stage, avg_losses, wer, improved, epoch_time):
        parts = [f"Epoch {epoch:3d}/{self.total_epochs}",
                 f"stage {stage}",
                 f"lr {self._current_lr():.1e}"]

        loss_parts = []
        if "ctc_g" in avg_losses:
            loss_parts.append(f"ctc_g {avg_losses['ctc_g']:.4f}")
        if "ctc_v" in avg_losses:
            loss_parts.append(f"ctc_v {avg_losses['ctc_v']:.4f}")
        if "seg" in avg_losses:
            loss_parts.append(f"seg {avg_losses['seg']:.4f}")
        if "total" in avg_losses:
            loss_parts.append(f"total {avg_losses['total']:.4f}")

        line = " | ".join(parts) + "  ||  " + "  ".join(loss_parts)

        if wer is not None:
            marker = "  <-- best" if improved else ""
            line += f"  ||  WER {wer:6.2f}%{marker}"

        line += f"  ({_fmt_time(epoch_time)})"
        print(line)

    def _print_final_summary(self, total_time: float):
        print()
        print(_box_title("Training complete"))
        print(f"  total time      : {_fmt_time(total_time)}")
        if self.best_epoch is not None:
            print(f"  best WER        : {self.best_wer:.2f}%  (epoch {self.best_epoch})")
        else:
            print("  best WER        : (no evaluation ran)")
        print(_hr("="))

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, epoch: int, wer: float):
        path = os.path.join(self.ckpt_dir, f"smkd_best_ep{epoch}_wer{wer:.2f}.pt")
        torch.save({
            "epoch": epoch, "wer": wer,
            # Always save the unwrapped model's state_dict, so checkpoints
            # load correctly regardless of DataParallel.
            "model": self.raw_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "vocab": self.vocab,
            "history": self.history,
        }, path)
        print(f"           >> checkpoint saved: {os.path.basename(path)}")

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_history(self, save_path: Optional[str] = None, show: bool = True):
        """
        Plots WER over epochs (and training losses) from `self.history`.

        Args:
            save_path: if given, saves the figure to this path (e.g. "wer.png")
            show: if True, calls plt.show() (useful in notebooks)

        Returns:
            the matplotlib Figure object.
        """
        import matplotlib.pyplot as plt

        epochs = self.history["epoch"]
        wer_epochs = [e for e, w in zip(epochs, self.history["wer"]) if w is not None]
        wer_values = [w for w in self.history["wer"] if w is not None]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

        # --- Left: WER over epochs ---
        ax = axes[0]
        if wer_values:
            ax.plot(wer_epochs, wer_values, marker="o", color="tab:red", label="Dev WER")
            if self.best_epoch is not None:
                ax.scatter([self.best_epoch], [self.best_wer], color="gold",
                           edgecolor="black", zorder=5, s=80,
                           label=f"Best ({self.best_wer:.2f}% @ ep {self.best_epoch})")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("WER (%)")
        ax.set_title("Word Error Rate over epochs")
        ax.legend()
        ax.grid(alpha=0.3)

        # Shade stage regions
        self._shade_stages(ax)

        # --- Right: training losses over epochs ---
        ax = axes[1]
        for key, label, color in [
            ("ctc_g", "CTC (contextual)", "tab:blue"),
            ("ctc_v", "CTC (visual)", "tab:orange"),
            ("seg", "GSBA segmentation", "tab:green"),
            ("total", "Total loss", "tab:gray"),
        ]:
            vals = self.history[key]
            xs = [e for e, v in zip(epochs, vals) if v is not None]
            ys = [v for v in vals if v is not None]
            if ys:
                ax.plot(xs, ys, label=label, color=color, alpha=0.85)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training losses over epochs")
        ax.legend()
        ax.grid(alpha=0.3)
        self._shade_stages(ax)

        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=120, bbox_inches="tight")
            print(f"Saved plot to {save_path}")
        if show:
            plt.show()

        return fig

    def _shade_stages(self, ax):
        """Adds light vertical shading for stage 1/2/3 regions."""
        stage_colors = {1: "tab:blue", 2: "tab:green", 3: "tab:purple"}
        bounds = [0.5, self.stage1_end + 0.5, self.stage2_end + 0.5, self.total_epochs + 0.5]
        for stage in (1, 2, 3):
            ax.axvspan(bounds[stage - 1], bounds[stage], color=stage_colors[stage], alpha=0.05)

    def print_history_table(self):
        """Prints a compact table of epoch / stage / losses / WER."""
        print(_box_title("Training history"))
        header = f"{'epoch':>5} {'stage':>5} {'ctc_g':>8} {'ctc_v':>8} {'seg':>8} {'total':>8} {'WER':>8}"
        print(header)
        print(_hr())
        for i, ep in enumerate(self.history["epoch"]):
            def fmt(key):
                v = self.history[key][i]
                return f"{v:8.4f}" if v is not None else f"{'--':>8}"

            wer = self.history["wer"][i]
            wer_str = f"{wer:7.2f}%" if wer is not None else f"{'--':>8}"

            print(f"{ep:5d} {self.history['stage'][i]:5d} "
                  f"{fmt('ctc_g')} {fmt('ctc_v')} {fmt('seg')} {fmt('total')} {wer_str}")
        print(_hr("="))