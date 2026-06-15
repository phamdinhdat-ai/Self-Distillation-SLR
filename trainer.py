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

import copy
import glob
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
            cache_dir=cfg.get("cache_dir", None),
            temporal_aug_enabled=cfg.get("temporal_aug_enabled", False),
            temporal_mask_prob=cfg.get("temporal_mask_prob", 0.3),
            temporal_mask_max_ratio=cfg.get("temporal_mask_max_ratio", 0.15),
            temporal_jitter_range=cfg.get("temporal_jitter_range", 0.0),
        )
        val_ds = PhoenixDataset(
            root=cfg.get("data_root", "./data/phoenix"),
            split="dev",
            vocab=train_ds.vocab,
            img_size=cfg.get("img_size", 224),
            max_frames=cfg.get("max_frames", 300),
            synthetic=cfg.get("synthetic", True),
            cache_dir=cfg.get("cache_dir", None),
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
            backbone=cfg.get("backbone", "resnet18"),
            temporal_conv_type=cfg.get("temporal_conv_type", "standard"),
            multi_scale_temporal=cfg.get("multi_scale_temporal", False),
            multi_scale_dilation_rates=cfg.get("multi_scale_dilation_rates", [1, 2]),
            use_tsm=cfg.get("use_tsm", False),
            pretrained_backbone=cfg.get("pretrained_backbone", False),
            freeze_backbone=cfg.get("freeze_backbone", False),
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
            # Enhancement flags
            feature_alignment_enabled=cfg.get("feature_alignment_enabled", False),
            feature_alignment_weight=cfg.get("feature_alignment_weight", 0.1),
            feature_alignment_mode=cfg.get("feature_alignment_mode", "cosine"),
            entropy_reg_enabled=cfg.get("entropy_reg_enabled", False),
            entropy_reg_weight=cfg.get("entropy_reg_weight", 0.01),
            focal_ctc_enabled=cfg.get("focal_ctc_enabled", False),
            focal_ctc_gamma=cfg.get("focal_ctc_gamma", 2.0),
            gsba_confidence_weight=cfg.get("gsba_confidence_weight", False),
            self_distill_enabled=cfg.get("self_distill_enabled", False),
            kd_temperature=cfg.get("kd_temperature", 4.0),
            kd_weight=cfg.get("kd_weight", 0.5),
        )

        self.optimizer = Adam(self.model.parameters(), lr=cfg.get("lr", 1e-4))
        self.scheduler = MultiStepLR(
            self.optimizer, milestones=cfg.get("lr_milestones", [40, 60, 80]),
            gamma=cfg.get("lr_gamma", 0.5),
        )

        # ---- AMP (mixed precision) ----
        self.use_amp = cfg.get("use_amp", False) and self.device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        self.grad_accum_steps = max(1, cfg.get("grad_accum_steps", 1))

        # ---- EMA (exponential moving average of weights) ----
        self.use_ema = cfg.get("use_ema", False)
        self.ema_decay = cfg.get("ema_decay", 0.999)
        self.ema_model = None
        if self.use_ema:
            self.ema_model = copy.deepcopy(self.raw_model)
            for p in self.ema_model.parameters():
                p.requires_grad_(False)

        # ---- Curriculum α ----
        self.alpha_schedule = cfg.get("alpha_schedule", "fixed")
        self.alpha_start = cfg.get("alpha_start", 0.1)
        self.alpha_end = cfg.get("alpha_end", 0.5)

        # ---- Smooth stage transitions ----
        self.smooth_transitions = cfg.get("smooth_transitions", False)
        self.smooth_transition_epochs = cfg.get("smooth_transition_epochs", 5)

        # ---- Self-distillation teacher ----
        self.teacher_model = None
        if cfg.get("self_distill_enabled", False) and cfg.get("self_distill_teacher_ckpt", ""):
            self._load_teacher(cfg["self_distill_teacher_ckpt"])

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

        # ---- Per-step timing profiler ----
        # Cumulative seconds per phase, reset each epoch
        self.step_timers = {
            "data_xfer": 0.0,     # batch → GPU
            "forward": 0.0,       # model(frames)
            "gsba": 0.0,          # GSBA algorithm (stage 2)
            "loss": 0.0,          # criterion.forward()
            "backward": 0.0,      # loss.backward() + optimizer.step()
            "total": 0.0,
        }
        # Per-epoch history for each timer
        for key in list(self.step_timers.keys()):
            self.history[f"t_{key}"] = []

        # ---- Stage checkpointing ----
        self.save_stage_ckpts = cfg.get("save_stage_checkpoints", True)
        self.resume_from_stage = cfg.get("resume_from_stage", 0)  # 0 = from scratch
        self.resume_ckpt_path = cfg.get("resume_ckpt_path", "")

        self._print_setup_summary()

    # ------------------------------------------------------------------
    # Setup summary
    # ------------------------------------------------------------------

    def _print_setup_summary(self):
        # ---- Parameter stats ----
        def _param_stats(model, name=""):
            total = sum(p.numel() for p in model.parameters())
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            frozen = total - trainable
            # Size in MB (float32)
            size_mb = total * 4 / (1024 * 1024)
            return total, trainable, frozen, size_mb

        main_total, main_trainable, main_frozen, main_mb = _param_stats(self.raw_model)
        ema_total = ema_trainable = ema_frozen = ema_mb = 0
        if self.use_ema and self.ema_model is not None:
            ema_total, ema_trainable, ema_frozen, ema_mb = _param_stats(self.ema_model)

        teacher_total = teacher_trainable = teacher_frozen = teacher_mb = 0
        if self.teacher_model is not None:
            teacher_total, teacher_trainable, teacher_frozen, teacher_mb = _param_stats(self.teacher_model)

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
        if self.cfg.get("cache_dir"):
            print(f"  frame cache     : {self.cfg['cache_dir']}")

        # ---- Model parameter summary ----
        print(f"  ── Model parameters ──")
        print(f"  main model        : {main_total:>10,} total  │  "
              f"{main_trainable:>10,} trainable  │  "
              f"{main_frozen:>10,} frozen  │  {main_mb:.1f} MB")
        if self.use_ema and self.ema_model is not None:
            print(f"  EMA model         : {ema_total:>10,} total  │  "
                  f"{ema_trainable:>10,} trainable  │  "
                  f"{ema_frozen:>10,} frozen  │  {ema_mb:.1f} MB  (all frozen)")
        if self.teacher_model is not None:
            print(f"  KD teacher model  : {teacher_total:>10,} total  │  "
                  f"{teacher_trainable:>10,} trainable  │  "
                  f"{teacher_frozen:>10,} frozen  │  {teacher_mb:.1f} MB  (all frozen)")

        # Total across all models
        grand_total = main_total + ema_total + teacher_total
        grand_mb = grand_total * 4 / (1024 * 1024)
        if ema_total > 0 or teacher_total > 0:
            print(f"  ── combined       : {grand_total:>10,} total  │  "
                  f"{main_trainable:>10,} active   │  {grand_mb:.1f} MB total")

        print(f"  d_model          : {self.raw_model.d_model}")
        backbone_name = self.cfg.get("backbone", "resnet18")
        backbone_note = backbone_name
        if self.cfg.get("pretrained_backbone", False):
            backbone_note += " (ImageNet pretrained)"
        if self.cfg.get("freeze_backbone", False):
            backbone_note += " [FROZEN]"
        print(f"  backbone         : {backbone_note}")
        print(f"  temporal conv    : {self.cfg.get('temporal_conv_type', 'standard')}")
        print(f"  total epochs     : {self.total_epochs}")
        print(f"  stage schedule   :")
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
        if self.save_stage_ckpts:
            print(f"  stage ckpts     : saved at each stage boundary")
        if self.resume_from_stage > 0:
            ckpt_info = self.resume_ckpt_path or "auto-find"
            print(f"  resume          : from stage {self.resume_from_stage} ({ckpt_info})")

        # Enhancement features summary
        features = []
        if self.use_amp:
            features.append(f"AMP (grad_accum={self.grad_accum_steps})")
        if self.use_ema:
            features.append(f"EMA (decay={self.ema_decay})")
        if self.alpha_schedule != "fixed":
            features.append(f"α curriculum: {self.alpha_schedule} ({self.alpha_start}→{self.alpha_end})")
        if self.smooth_transitions:
            features.append(f"smooth transitions ({self.smooth_transition_epochs}ep)")
        if self.criterion.fa_enabled:
            features.append(f"VAC align ({self.criterion.fa_mode}, w={self.criterion.fa_weight})")
        if self.criterion.entropy_enabled:
            features.append(f"entropy reg (w={self.criterion.entropy_weight})")
        if self.criterion.confidence_weight:
            features.append("confidence-weighted GSBA")
        if self.teacher_model is not None:
            features.append("self-distillation KD")
        if self.cfg.get("temporal_aug_enabled", False):
            features.append("temporal augmentation")
        if features:
            print(f"  enhancements    :")
            for feat in features:
                print(f"      - {feat}")
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

    def _get_alpha(self, epoch: int, stage: int) -> float:
        """Get α weight, potentially following a curriculum schedule."""
        if self.alpha_schedule == "fixed" or stage != 1:
            return self.cfg.get("alpha", 0.5)

        # Only schedule α during stage 1
        progress = min(1.0, epoch / max(self.stage1_end, 1))

        if self.alpha_schedule == "linear":
            return self.alpha_start + (self.alpha_end - self.alpha_start) * progress
        elif self.alpha_schedule == "cosine":
            return self.alpha_end - (self.alpha_end - self.alpha_start) * (1 + torch.cos(
                torch.tensor(progress * 3.14159))) / 2
        return self.cfg.get("alpha", 0.5)

    def _get_smooth_factor(self, epoch: int) -> float:
        """
        Returns a smooth transition factor λ ∈ [0, 1].
        λ=1 → full-stage behavior; λ=0 → full next-stage behavior.
        Used at stage boundaries (end of stage 1 → stage 2, end of stage 2 → stage 3).
        """
        if not self.smooth_transitions or self.smooth_transition_epochs <= 0:
            return 1.0

        # Check if we're in a transition zone
        # Stage 1 → 2 transition (last N epochs of stage 1)
        if self.stage1_end - self.smooth_transition_epochs < epoch <= self.stage1_end:
            progress = (self.stage1_end - epoch) / self.smooth_transition_epochs
            return progress  # 1 at start of transition → 0 at end
        # Stage 2 → 3 transition
        if self.stage2_end - self.smooth_transition_epochs < epoch <= self.stage2_end:
            progress = (self.stage2_end - epoch) / self.smooth_transition_epochs
            return progress  # 1 at start → 0 at end
        return 1.0

    def _update_ema(self):
        """Update EMA model parameters after an optimizer step."""
        if not self.use_ema or self.ema_model is None:
            return
        with torch.no_grad():
            for ema_p, raw_p in zip(self.ema_model.parameters(), self.raw_model.parameters()):
                ema_p.data.mul_(self.ema_decay).add_(raw_p.data, alpha=1.0 - self.ema_decay)

    def _set_stage(self, stage: int):
        """Sets the training stage on the underlying (unwrapped) model."""
        self.raw_model.set_stage(stage)

    def _load_teacher(self, ckpt_path: str):
        """Load a pre-trained SMKD model for self-distillation KD."""
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.teacher_model = SMKD(
            num_classes=len(self.vocab),
            d_model=self.cfg.get("d_model", 512),
            hidden_size=self.cfg.get("hidden_size", 512),
            backbone=self.cfg.get("backbone", "resnet18"),
            temporal_conv_type=self.cfg.get("temporal_conv_type", "standard"),
            multi_scale_temporal=self.cfg.get("multi_scale_temporal", False),
            multi_scale_dilation_rates=self.cfg.get("multi_scale_dilation_rates", [1, 2]),
            use_tsm=self.cfg.get("use_tsm", False),
            pretrained_backbone=False,  # teacher loads its own weights
            freeze_backbone=False,
        ).to(self.device)
        self.teacher_model.load_state_dict(ckpt["model"])
        self.teacher_model.eval()
        for p in self.teacher_model.parameters():
            p.requires_grad_(False)
        print(f"  [*] Loaded self-distillation teacher from: {ckpt_path}")

    # ------------------------------------------------------------------
    # Train step
    # ------------------------------------------------------------------

    def _train_step(self, batch: dict, stage: int, epoch: int = 1, batch_idx: int = 0):
        t0 = time.time()

        # --- Data transfer ---
        frames = batch["frames"].to(self.device, non_blocking=True)
        labels = batch["labels"].to(self.device, non_blocking=True)
        input_lengths = batch["input_lengths"].to(self.device, non_blocking=True)
        target_lengths = batch["target_lengths"].to(self.device, non_blocking=True)
        gloss_seqs = batch["gloss_seqs"]
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t_data = time.time()

        # --- Forward pass ---
        with torch.cuda.amp.autocast() if self.use_amp else torch.no_grad() if False else torch.enable_grad():
            out = self.model(frames)
            logits_v, logits_g, gcf, lvf = out["logits_v"], out["logits_g"], out["gcf"], out["lvf"]

        T_prime = logits_v.size(1)
        adj_lengths = torch.full((frames.size(0),), T_prime, dtype=torch.long, device=self.device)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t_forward = time.time()

        # --- GSBA (stage 2) ---
        seg_labels = None
        seg_margins = None
        if stage == 2:
            with torch.no_grad():
                spike_labels = logits_g.argmax(dim=-1)
                w_n = F.normalize(self.raw_model.shared_classifier.weight, p=2, dim=-1)
                need_margins = self.criterion.confidence_weight
                if need_margins:
                    seg_labels, seg_margins = batch_gsba(
                        gcf=gcf.detach(), classifier_weights=w_n.detach(),
                        spike_labels=spike_labels, gloss_seqs=gloss_seqs, d=self.gsba_d,
                        return_margins=True,
                    )
                else:
                    seg_labels = batch_gsba(
                        gcf=gcf.detach(), classifier_weights=w_n.detach(),
                        spike_labels=spike_labels, gloss_seqs=gloss_seqs, d=self.gsba_d,
                    )
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t_gsba = time.time()

        # --- Teacher logits for self-distillation ---
        teacher_logits_g = None
        if self.teacher_model is not None:
            with torch.no_grad():
                teacher_out = self.teacher_model(frames)
                teacher_logits_g = teacher_out["logits_g"].detach()

        # --- Set effective α for curriculum + smooth transitions ---
        self.criterion.alpha = self._get_alpha(epoch, stage) * self._get_smooth_factor(epoch)

        # --- Loss computation ---
        loss, loss_dict = self.criterion(
            logits_v=logits_v, logits_g=logits_g, targets=labels,
            input_lengths=adj_lengths, target_lengths=target_lengths,
            stage=stage, seg_labels=seg_labels, seg_margins=seg_margins,
            lvf=lvf, gcf=gcf,
            teacher_logits_g=teacher_logits_g,
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t_loss = time.time()

        # --- Backward + optimizer step ---
        if self.use_amp:
            self.scaler.scale(loss / self.grad_accum_steps).backward()
        else:
            (loss / self.grad_accum_steps).backward()

        if (batch_idx + 1) % self.grad_accum_steps == 0:
            if self.use_amp:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.optimizer.step()
            self.optimizer.zero_grad()
            self._update_ema()

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t_backward = time.time()

        # --- Accumulate timing ---
        # Only count backward time for steps that actually step the optimizer
        n_accum = self.grad_accum_steps
        self.step_timers["data_xfer"] += (t_data - t0)
        self.step_timers["forward"] += (t_forward - t_data)
        self.step_timers["gsba"] += (t_gsba - t_forward)
        self.step_timers["loss"] += (t_loss - t_gsba)
        self.step_timers["backward"] += (t_backward - t_loss)
        self.step_timers["total"] += (t_backward - t0)

        return loss_dict

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self) -> float:
        # Use EMA model for evaluation if available
        eval_model = self.ema_model if self.use_ema and self.ema_model is not None else self.raw_model
        eval_model.eval()
        all_hyps, all_refs = [], []

        # Temporarily swap if using DataParallel
        original_model = self.model
        if self.use_ema and self.ema_model is not None:
            if self.use_data_parallel:
                self.model = nn.DataParallel(self.ema_model)
            else:
                self.model = self.ema_model

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

        # Restore original model
        self.model = original_model
        eval_model.train()
        return compute_wer(all_hyps, all_refs)

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self):
        print(_box_title("Training started"))
        train_start = time.time()

        # Resume from stage checkpoint if requested
        start_epoch = 1
        if self.resume_from_stage > 0 and self.resume_ckpt_path:
            resume_ep = self._load_stage_checkpoint(self.resume_ckpt_path)
            start_epoch = resume_ep + 1
        elif self.resume_from_stage > 0:
            # Auto-find stage checkpoint
            for stage in range(self.resume_from_stage - 1, 0, -1):
                pattern = os.path.join(self.ckpt_dir, f"smkd_stage{stage}_ep*.pt")
                matches = sorted(glob.glob(pattern))
                if matches:
                    resume_ep = self._load_stage_checkpoint(matches[-1])
                    start_epoch = resume_ep + 1
                    break
            else:
                print(f"  [!] No stage checkpoint found for resume_from_stage={self.resume_from_stage}, starting from scratch")

        for epoch in range(start_epoch, self.total_epochs + 1):
            epoch_start = time.time()

            stage = self._current_stage(epoch)
            self._maybe_update_gsba_d(epoch)

            if self.raw_model.stage != stage:
                self._set_stage(stage)
                # Save stage checkpoint at boundary (end of previous stage)
                if self.save_stage_ckpts and epoch > 1:
                    self._save_stage_checkpoint(stage - 1, epoch - 1)
                print()
                print(_box_title(f"Stage {stage}: {STAGE_NAMES[stage]}"))
                if stage == 3:
                    self.optimizer = Adam(self.model.parameters(), lr=4e-6)

            # Reset step timers for this epoch
            for k in self.step_timers:
                self.step_timers[k] = 0.0

            # Update α on criterion for the current epoch
            self.criterion.alpha = self._get_alpha(epoch, stage)

            # --- run one epoch ---
            epoch_losses = {}
            for batch_idx, batch in enumerate(self.train_loader):
                loss_dict = self._train_step(batch, stage, epoch=epoch, batch_idx=batch_idx)
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
            for ek in ("align", "entropy", "kd"):
                if ek not in self.history:
                    self.history[ek] = []
                self.history[ek].append(avg_losses.get(ek))
            # Per-step timing history
            for tk in self.step_timers:
                self.history[f"t_{tk}"].append(self.step_timers[tk])

            # --- pretty print ---
            self._print_epoch_line(epoch, stage, avg_losses, wer, improved, epoch_time)

            # Timing breakdown every eval or every 10 epochs
            if run_eval or epoch % 10 == 0:
                self._print_timing_breakdown(epoch)

            if stage < 3:
                self.scheduler.step()

        # Save final stage checkpoint
        final_stage = self._current_stage(self.total_epochs)
        if self.save_stage_ckpts:
            self._save_stage_checkpoint(final_stage, self.total_epochs)

        total_time = time.time() - train_start
        self._print_final_summary(total_time)

    # ------------------------------------------------------------------
    # Pretty printing
    # ------------------------------------------------------------------

    def _print_epoch_line(self, epoch, stage, avg_losses, wer, improved, epoch_time):
        alpha = self.criterion.alpha
        alpha_str = f"α={alpha:.2f}" if self.alpha_schedule != "fixed" or alpha != 0.5 else ""
        parts = [f"Epoch {epoch:3d}/{self.total_epochs}",
                 f"stage {stage}",
                 f"lr {self._current_lr():.1e}"]
        if alpha_str:
            parts.append(alpha_str)

        loss_parts = []
        if "ctc_g" in avg_losses:
            loss_parts.append(f"ctc_g {avg_losses['ctc_g']:.4f}")
        if "ctc_v" in avg_losses:
            loss_parts.append(f"ctc_v {avg_losses['ctc_v']:.4f}")
        if "seg" in avg_losses:
            loss_parts.append(f"seg {avg_losses['seg']:.4f}")
        for ek in ("align", "entropy", "kd"):
            if ek in avg_losses and avg_losses[ek] is not None and avg_losses[ek] != 0:
                loss_parts.append(f"{ek} {avg_losses[ek]:.4f}")
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
        n_params = sum(p.numel() for p in self.raw_model.parameters())
        n_trainable = sum(p.numel() for p in self.raw_model.parameters() if p.requires_grad)
        print(f"  model params    : {n_params:,} total  |  {n_trainable:,} trainable  |  "
              f"{n_params - n_trainable:,} frozen")

        # Cache statistics
        train_ds = self.train_loader.dataset
        if hasattr(train_ds, "cache_stats") and train_ds.cache_dir:
            stats = train_ds.cache_stats()
            train_label = "train"
            val_ds = self.val_loader.dataset
            val_stats = val_ds.cache_stats() if hasattr(val_ds, "cache_stats") else None
            print(f"  frame cache     : {train_label} {stats['hits']}/{stats['total']} hits "
                  f"({stats['hit_rate']:.0%})", end="")
            if val_stats:
                print(f"  |  dev {val_stats['hits']}/{val_stats['total']} hits "
                      f"({val_stats['hit_rate']:.0%})", end="")
            print()
        print(_hr("="))

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, epoch: int, wer: float):
        path = os.path.join(self.ckpt_dir, f"smkd_best_ep{epoch}_wer{wer:.2f}.pt")
        self._write_checkpoint(path, epoch, wer)
        print(f"           >> checkpoint saved: {os.path.basename(path)}")

    def _save_stage_checkpoint(self, stage: int, epoch: int):
        """Save a checkpoint at the end of a training stage for later resume."""
        path = os.path.join(self.ckpt_dir, f"smkd_stage{stage}_ep{epoch}.pt")
        self._write_checkpoint(path, epoch, None)
        print(f"           >> stage {stage} checkpoint saved: {os.path.basename(path)}")

    def _write_checkpoint(self, path: str, epoch: int, wer):
        ckpt = {
            "epoch": epoch, "wer": wer,
            "stage": self.raw_model.stage,
            "model": self.raw_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "vocab": self.vocab,
            "history": self.history,
        }
        if self.use_ema and self.ema_model is not None:
            ckpt["ema_model"] = self.ema_model.state_dict()
        if self.use_amp and self.scaler is not None:
            ckpt["scaler"] = self.scaler.state_dict()
        torch.save(ckpt, path)

    def _load_stage_checkpoint(self, ckpt_path: str):
        """Resume training from a previously saved stage checkpoint."""
        print(f"\n  [*] Resuming from stage checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.raw_model.load_state_dict(ckpt["model"])
        self.raw_model.set_stage(ckpt.get("stage", self.raw_model.stage))
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.history = ckpt.get("history", self.history)

        if self.use_ema and self.ema_model is not None and "ema_model" in ckpt:
            self.ema_model.load_state_dict(ckpt["ema_model"])
        if self.use_amp and self.scaler is not None and "scaler" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler"])

        resume_epoch = ckpt["epoch"]
        resume_wer = ckpt.get("wer")
        print(f"  [*] Resumed at epoch {resume_epoch}, stage {self.raw_model.stage}"
              + (f", WER={resume_wer:.2f}%" if resume_wer else ""))
        return resume_epoch

    def _print_timing_breakdown(self, epoch: int):
        """Print per-step timing breakdown for the current epoch."""
        total = self.step_timers["total"]
        if total < 0.01:
            return

        def pct(key):
            return 100.0 * self.step_timers[key] / total

        print(f"  ⏱  timing epoch {epoch:3d}: "
              f"data={self.step_timers['data_xfer']:5.1f}s ({pct('data_xfer'):4.1f}%)  "
              f"fwd={self.step_timers['forward']:5.1f}s ({pct('forward'):4.1f}%)  "
              f"gsba={self.step_timers['gsba']:5.1f}s ({pct('gsba'):4.1f}%)  "
              f"loss={self.step_timers['loss']:5.1f}s ({pct('loss'):4.1f}%)  "
              f"bwd={self.step_timers['backward']:5.1f}s ({pct('backward'):4.1f}%)")

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