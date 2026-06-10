"""
Three-stage Trainer for SMKD.

Stage schedule (default, adjustable via config):
  epochs  1–30   → Stage 1: synchronous training (shared classifier)
  epochs 31–90   → Stage 2: gloss segmentation (GSBA, d grows every 20 ep)
  epochs 91–100  → Stage 3: decouple training   (independent classifiers)
"""

import os
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

from models.smkd  import SMKD
from models.losses import SMKDLoss, batch_gsba
from data.dataset  import PhoenixDataset, collate_fn, Vocabulary
from metrics       import compute_wer


# ---------------------------------------------------------------------------
# Greedy CTC decoder
# ---------------------------------------------------------------------------

def greedy_ctc_decode(logits: torch.Tensor, blank: int = 0) -> List[List[int]]:
    """
    Args:
        logits: (B, T', C+1)
    Returns:
        list of B decoded sequences (no blanks, no repeats)
    """
    preds = logits.argmax(dim=-1).cpu().tolist()   # (B, T')
    results = []
    for seq in preds:
        decoded = []
        prev = blank
        for tok in seq:
            if tok != blank and tok != prev:
                decoded.append(tok)
            prev = tok
        results.append(decoded)
    return results


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    def __init__(self, cfg: dict):
        self.cfg   = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ---- Data ----
        print("Loading datasets ...")
        train_ds = PhoenixDataset(
            root      = cfg.get("data_root", "./data/phoenix"),
            split     = "train",
            img_size  = cfg.get("img_size", 224),
            max_frames= cfg.get("max_frames", 300),
            synthetic = cfg.get("synthetic", True),
        )
        val_ds = PhoenixDataset(
            root      = cfg.get("data_root", "./data/phoenix"),
            split     = "dev",
            vocab     = train_ds.vocab,
            img_size  = cfg.get("img_size", 224),
            max_frames= cfg.get("max_frames", 300),
            synthetic = cfg.get("synthetic", True),
        )
        self.vocab = train_ds.vocab
        self.train_loader = DataLoader(
            train_ds,
            batch_size  = cfg.get("batch_size", 2),
            shuffle     = True,
            collate_fn  = collate_fn,
            num_workers = cfg.get("num_workers", 0),
        )
        self.val_loader = DataLoader(
            val_ds,
            batch_size  = cfg.get("batch_size", 2),
            shuffle     = False,
            collate_fn  = collate_fn,
            num_workers = cfg.get("num_workers", 0),
        )

        # ---- Model ----
        num_classes = len(self.vocab)
        self.model = SMKD(
            num_classes  = num_classes,
            d_model      = cfg.get("d_model", 512),
            hidden_size  = cfg.get("hidden_size", 512),
        ).to(self.device)

        # ---- Loss ----
        self.criterion = SMKDLoss(
            blank          = 0,
            alpha          = cfg.get("alpha", 0.5),
            label_smoothing= cfg.get("label_smoothing", 0.2),
        )

        # ---- Optimiser (stages 1 & 2) ----
        self.optimizer = Adam(self.model.parameters(), lr=cfg.get("lr", 1e-4))
        self.scheduler = MultiStepLR(
            self.optimizer,
            milestones = cfg.get("lr_milestones", [40, 60, 80]),
            gamma      = 0.5,
        )

        # ---- Stage schedule ----
        self.stage1_end  = cfg.get("stage1_end", 30)
        self.stage2_end  = cfg.get("stage2_end", 90)
        self.total_epochs= cfg.get("epochs", 100)

        # GSBA params
        self.gsba_d            = 1
        self.gsba_update_every = cfg.get("gsba_update_every", 10)
        self.seg_labels_cache  = None   # (B, T') refreshed every N epochs

        # Book-keeping
        self.best_wer  = float("inf")
        self.ckpt_dir  = cfg.get("ckpt_dir", "./checkpoints")
        os.makedirs(self.ckpt_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Stage helpers
    # ------------------------------------------------------------------

    def _current_stage(self, epoch: int) -> int:
        if epoch <= self.stage1_end:
            return 1
        elif epoch <= self.stage2_end:
            return 2
        else:
            return 3

    def _maybe_update_gsba_d(self, epoch: int):
        """Increase GSBA radius d by 1 every 20 epochs (after stage 2 starts)."""
        if epoch > self.stage1_end:
            self.gsba_d = 1 + (epoch - self.stage1_end - 1) // 20

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def _train_step(self, batch: dict, stage: int):
        frames         = batch["frames"].to(self.device)          # (B, T, 3, H, W)
        labels         = batch["labels"].to(self.device)
        input_lengths  = batch["input_lengths"].to(self.device)
        target_lengths = batch["target_lengths"].to(self.device)
        gloss_seqs     = batch["gloss_seqs"]

        out = self.model(frames)
        logits_v = out["logits_v"]    # (B, T', C+1)
        logits_g = out["logits_g"]
        gcf      = out["gcf"]         # (B, T', d)

        # Adjusted input lengths after temporal downsampling (÷2)
        T_prime = logits_v.size(1)
        adj_lengths = torch.clamp(input_lengths // 2, max=T_prime)

        # Generate segment labels for stage 2
        seg_labels = None
        if stage == 2:
            with torch.no_grad():
                # Greedy pseudo-labels from contextual module
                spike_labels = logits_g.argmax(dim=-1)      # (B, T')
                w = self.model.shared_classifier.weight      # (C+1, d)
                w_n = F.normalize(w, p=2, dim=-1)
                seg_labels = batch_gsba(
                    gcf        = gcf.detach(),
                    classifier_weights = w_n.detach(),
                    spike_labels       = spike_labels,
                    gloss_seqs         = gloss_seqs,
                    d                  = self.gsba_d,
                )

        loss, loss_dict = self.criterion(
            logits_v       = logits_v,
            logits_g       = logits_g,
            targets        = labels,
            input_lengths  = adj_lengths,
            target_lengths = target_lengths,
            stage          = stage,
            seg_labels     = seg_labels,
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
            frames         = batch["frames"].to(self.device)
            input_lengths  = batch["input_lengths"].to(self.device)
            target_lengths = batch["target_lengths"].to(self.device)
            labels         = batch["labels"]
            gloss_seqs     = batch["gloss_seqs"]

            out      = self.model(frames)
            T_prime  = out["logits_g"].size(1)
            adj_len  = torch.clamp(input_lengths // 2, max=T_prime)

            hyps = greedy_ctc_decode(out["logits_g"])
            # Rebuild reference sequences from flat tensor
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
        print(f"Training on {self.device}  |  vocab size: {len(self.vocab)}")

        for epoch in range(1, self.total_epochs + 1):
            stage = self._current_stage(epoch)
            self._maybe_update_gsba_d(epoch)

            # Switch stage in model
            if self.model.stage != stage:
                self.model.set_stage(stage)
                print(f"\n>>> Entering Stage {stage} at epoch {epoch} <<<\n")

                # Stage 3: reinitialise optimiser with lower LR
                if stage == 3:
                    self.optimizer = Adam(
                        self.model.parameters(), lr=4e-6
                    )

            # ---- Epoch loop ----
            epoch_losses = {}
            for batch in self.train_loader:
                loss_dict = self._train_step(batch, stage)
                for k, v in loss_dict.items():
                    epoch_losses[k] = epoch_losses.get(k, 0.0) + v

            n = len(self.train_loader)
            log = f"Epoch {epoch:3d} | stage={stage} | d={self.gsba_d}"
            for k, v in epoch_losses.items():
                log += f" | {k}={v/n:.4f}"

            # Evaluate every 5 epochs
            if epoch % 5 == 0 or epoch == self.total_epochs:
                wer = self.evaluate()
                log += f" | WER={wer:.2f}%"
                if wer < self.best_wer:
                    self.best_wer = wer
                    self._save_checkpoint(epoch, wer)

            print(log)

            if stage < 3:
                self.scheduler.step()

        print(f"\nTraining complete. Best WER: {self.best_wer:.2f}%")

    # ------------------------------------------------------------------
    def _save_checkpoint(self, epoch: int, wer: float):
        path = os.path.join(self.ckpt_dir, f"smkd_best_ep{epoch}_wer{wer:.2f}.pt")
        torch.save({
            "epoch":      epoch,
            "wer":        wer,
            "model":      self.model.state_dict(),
            "optimizer":  self.optimizer.state_dict(),
            "vocab":      self.vocab,
        }, path)
        print(f"  [✓] Checkpoint saved → {path}")
