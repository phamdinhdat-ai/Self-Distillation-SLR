"""
Loss functions and the Gloss Segment Boundary Assignment (GSBA) algorithm
for SMKD training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


# ---------------------------------------------------------------------------
# CTC Loss wrapper (handles variable-length sequences)
# ---------------------------------------------------------------------------

class CTCLoss(nn.Module):
    def __init__(self, blank: int = 0, zero_infinity: bool = True):
        super().__init__()
        self.ctc = nn.CTCLoss(blank=blank, reduction="mean", zero_infinity=zero_infinity)
        self.blank = blank

    def forward(
        self,
        logits: torch.Tensor,          # (B, T', C+1)
        targets: torch.Tensor,         # (sum_N,)  flat concatenated labels
        input_lengths: torch.Tensor,   # (B,)
        target_lengths: torch.Tensor,  # (B,)
    ) -> torch.Tensor:
        # nn.CTCLoss expects (T', B, C+1) log-probs
        log_probs = F.log_softmax(logits, dim=-1)           # (B, T', C+1)
        log_probs = log_probs.permute(1, 0, 2)              # (T', B, C+1)
        return self.ctc(log_probs, targets, input_lengths, target_lengths)


# ---------------------------------------------------------------------------
# Label-smoothed Cross-Entropy loss
# ---------------------------------------------------------------------------

class LabelSmoothingCE(nn.Module):
    def __init__(self, smoothing: float = 0.2, ignore_index: int = -1):
        super().__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (N, C)  – unnormalised scores
            targets: (N,)    – integer class labels; -1 entries are ignored
        """
        C = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)

        # One-hot with label smoothing
        with torch.no_grad():
            smooth_targets = torch.full_like(log_probs, self.smoothing / (C - 1))
            # Mask out ignored positions before scatter
            valid = targets != self.ignore_index
            if valid.sum() == 0:
                return logits.new_tensor(0.0)
            valid_targets = targets.clone()
            valid_targets[~valid] = 0          # placeholder, will be masked
            smooth_targets.scatter_(-1, valid_targets.unsqueeze(-1), 1.0 - self.smoothing)

        loss = -(smooth_targets * log_probs).sum(dim=-1)
        loss = loss[valid].mean()
        return loss


# ---------------------------------------------------------------------------
# GSBA – Gloss Segment Boundary Assignment
# ---------------------------------------------------------------------------

def gsba(
    gcf: torch.Tensor,              # (T', d)  Global Contextual Features for one sample
    classifier_weights: torch.Tensor,  # (C+1, d)  normalised weight matrix
    spike_labels: torch.Tensor,     # (T',)  CTC pseudo-labels (blank=0)
    gloss_seq: List[int],           # ordered gloss ids (no blanks) for this sample
    d: int = 1,                     # expansion radius
    blank: int = 0,
    return_margins: bool = False,
):
    """
    Produces per-frame gloss segment labels for one sequence.

    Algorithm:
      1. Identify anchor frames = frames whose CTC pseudo-label is non-blank.
      2. For each anchor frame t with class c_a:
         - Expand d steps backward (t-1 … t-d) and forward (t+1 … t+d).
         - Annotate the expanding frame with c_a if cosine-similarity between
           its GCF and w_{c_a} is the highest among all glosses in gloss_seq.
         - Stop expansion in that direction on first failure.
      3. Remaining frames stay labelled as blank (-1 → ignore in CE loss).

    Returns:
        seg_labels: (T',) long tensor;  -1 means "ignore"
        margins: (T',) float tensor (only if return_margins=True);
                 cosine-sim margin (winner - runner-up) at assigned positions, 0 elsewhere
    """
    T = gcf.size(0)
    seg_labels = torch.full((T,), -1, dtype=torch.long, device=gcf.device)
    margins = torch.zeros((T,), device=gcf.device) if return_margins else None

    # Normalise features and weights for cosine sim
    gcf_n = F.normalize(gcf, p=2, dim=-1)                  # (T', d)
    w_n   = F.normalize(classifier_weights, p=2, dim=-1)   # (C+1, d)

    # Only cosine sims for classes actually present in this sentence
    gloss_ids = torch.tensor(gloss_seq, device=gcf.device)  # (N_gloss,)
    w_gloss   = w_n[gloss_ids]                              # (N_gloss, d)

    def best_class_and_margin(t_idx: int):
        sims = gcf_n[t_idx] @ w_gloss.t()   # (N_gloss,)
        sorted_sims, sorted_idx = sims.sort(descending=True)
        best = int(sorted_idx[0].item())
        best_class_id = int(gloss_ids[best].item())
        if return_margins and len(sorted_sims) > 1:
            margin = sorted_sims[0] - sorted_sims[1]
        else:
            margin = 1.0
        return best_class_id, margin

    def best_class(t_idx: int) -> int:
        c, _ = best_class_and_margin(t_idx)
        return c

    # Anchor frames
    anchor_mask = (spike_labels != blank)
    anchor_indices = anchor_mask.nonzero(as_tuple=True)[0].tolist()

    for t in anchor_indices:
        c_a = int(spike_labels[t].item())
        seg_labels[t] = c_a
        if return_margins:
            _, m = best_class_and_margin(t)
            margins[t] = m

        # Expand backwards
        for dt in range(1, d + 1):
            ti = t - dt
            if ti < 0:
                break
            if seg_labels[ti] != -1:
                break
            bc, margin = best_class_and_margin(ti)
            if bc == c_a:
                seg_labels[ti] = c_a
                if return_margins:
                    margins[ti] = margin
            else:
                break

        # Expand forwards
        for dt in range(1, d + 1):
            ti = t + dt
            if ti >= T:
                break
            if seg_labels[ti] != -1:
                break
            bc, margin = best_class_and_margin(ti)
            if bc == c_a:
                seg_labels[ti] = c_a
                if return_margins:
                    margins[ti] = margin
            else:
                break

    if return_margins:
        return seg_labels, margins
    return seg_labels


def batch_gsba(
    gcf: torch.Tensor,                  # (B, T', d)
    classifier_weights: torch.Tensor,   # (C+1, d) – already normalised
    spike_labels: torch.Tensor,         # (B, T')
    gloss_seqs: List[List[int]],        # list of B gloss sequences
    d: int = 1,
    blank: int = 0,
    return_margins: bool = False,
):
    """Run GSBA for a full batch; returns (B, T') segment labels (and margins if requested)."""
    B, T = spike_labels.shape
    seg = torch.full((B, T), -1, dtype=torch.long, device=gcf.device)
    margins_out = torch.zeros((B, T), device=gcf.device) if return_margins else None
    for b in range(B):
        if return_margins:
            seg[b], margins_out[b] = gsba(
                gcf[b], classifier_weights,
                spike_labels[b], gloss_seqs[b],
                d=d, blank=blank, return_margins=True,
            )
        else:
            seg[b] = gsba(
                gcf[b], classifier_weights,
                spike_labels[b], gloss_seqs[b],
                d=d, blank=blank,
            )
    if return_margins:
        return seg, margins_out
    return seg


# ---------------------------------------------------------------------------
# SMKD Training Loss (all stages) – Enhanced version
# ---------------------------------------------------------------------------

class SMKDLoss(nn.Module):
    """
    Combines the per-stage training objectives with optional enhancements:

    Stage 1 (sync train):
        L = CTC(g) + α * CTC(v)
        [+ feature_alignment_weight * ||g_c - v_c||  if enabled]
        [+ entropy_reg_weight * (-H_blank)           if enabled]

    Stage 2 (gloss segmentation):
        L = CTC(g) + α * CE-LS(v, seg_labels)
        [+ feature_alignment_weight * ||g_c - v_c||  if enabled]
        [+ entropy_reg_weight * (-H_blank)           if enabled]
        [+ confidence weighting on CE-LS frames       if enabled]

    Stage 3 (decouple train):
        L = CTC(g)   [visual classifier detached; contextual only]
        [+ kd_weight * KD(g_student || g_teacher)    if self-distill enabled]
    """

    def __init__(
        self,
        blank: int = 0,
        alpha: float = 0.5,
        label_smoothing: float = 0.2,
        # --- Enhancements ---
        feature_alignment_enabled: bool = False,
        feature_alignment_weight: float = 0.1,
        feature_alignment_mode: str = "cosine",
        entropy_reg_enabled: bool = False,
        entropy_reg_weight: float = 0.01,
        focal_ctc_enabled: bool = False,
        focal_ctc_gamma: float = 2.0,
        gsba_confidence_weight: bool = False,
        # Self-distillation
        self_distill_enabled: bool = False,
        kd_temperature: float = 4.0,
        kd_weight: float = 0.5,
    ):
        super().__init__()
        self.blank = blank
        self.alpha = alpha
        self.label_smoothing = label_smoothing

        # Choose CTC variant
        if focal_ctc_enabled:
            self.ctc_loss = FocalCTCLoss(blank=blank, gamma=focal_ctc_gamma)
        else:
            self.ctc_loss = CTCLoss(blank=blank)

        self.ce_ls = LabelSmoothingCE(smoothing=label_smoothing, ignore_index=-1)

        # Enhancement flags
        self.fa_enabled = feature_alignment_enabled
        self.fa_weight = feature_alignment_weight
        self.fa_mode = feature_alignment_mode
        self.entropy_enabled = entropy_reg_enabled
        self.entropy_weight = entropy_reg_weight
        self.confidence_weight = gsba_confidence_weight
        self.self_distill = self_distill_enabled
        self.kd_temp = kd_temperature
        self.kd_weight = kd_weight

    def forward(
        self,
        logits_v: torch.Tensor,          # (B, T', C+1)
        logits_g: torch.Tensor,          # (B, T', C+1)
        targets: torch.Tensor,           # (sum targets,) flat
        input_lengths: torch.Tensor,     # (B,)
        target_lengths: torch.Tensor,    # (B,)
        stage: int,
        seg_labels: torch.Tensor = None, # (B, T')  required for stage 2
        seg_margins: torch.Tensor = None, # (B, T') optional GSBA margin weights
        lvf: torch.Tensor = None,        # (B, T', d) for feature alignment
        gcf: torch.Tensor = None,        # (B, T', d) for feature alignment
        teacher_logits_g: torch.Tensor = None,  # (B, T', C+1) self-distill teacher
    ) -> Tuple[torch.Tensor, dict]:

        losses = {}

        # CTC on contextual features (all stages)
        l_ctc_g = self.ctc_loss(logits_g, targets, input_lengths, target_lengths)
        total = l_ctc_g
        losses["ctc_g"] = l_ctc_g.item()

        if stage == 1:
            l_ctc_v = self.ctc_loss(logits_v, targets, input_lengths, target_lengths)
            losses["ctc_v"] = l_ctc_v.item()
            total = l_ctc_g + self.alpha * l_ctc_v

        elif stage == 2:
            # CE-LS on visual features with GSBA segment labels
            assert seg_labels is not None, "seg_labels required for stage 2"
            B, T_prime, C1 = logits_v.shape
            logits_v_flat   = logits_v.view(-1, C1)
            seg_labels_flat = seg_labels.view(-1)

            if self.confidence_weight and seg_margins is not None:
                # Confidence-weighted CE: multiply each frame's loss by its GSBA margin
                margins_flat = seg_margins.view(-1)
                valid_mask = seg_labels_flat != -1
                if valid_mask.sum() > 0:
                    l_seg = self._weighted_ce_ls(
                        logits_v_flat[valid_mask],
                        seg_labels_flat[valid_mask],
                        margins_flat[valid_mask],
                    )
                else:
                    l_seg = logits_v.new_tensor(0.0)
            else:
                l_seg = self.ce_ls(logits_v_flat, seg_labels_flat)

            losses["seg"] = l_seg.item()
            total = l_ctc_g + self.alpha * l_seg

        else:  # stage 3 – decouple
            pass  # total = l_ctc_g already set

        # ---- Optional: Feature alignment (VAC-style) ----
        if self.fa_enabled and lvf is not None and gcf is not None:
            # Compute mask: frames with non-ignore GSBA labels (stage 2) or all frames (stage 1)
            if seg_labels is not None:
                align_mask = (seg_labels != -1)   # (B, T')
            else:
                align_mask = torch.ones_like(logits_g[:, :, 0], dtype=torch.bool)
            l_align = feature_alignment_loss(lvf, gcf, align_mask, mode=self.fa_mode)
            losses["align"] = l_align.item()
            total = total + self.fa_weight * l_align

        # ---- Optional: Entropy regularization (anti-spike) ----
        if self.entropy_enabled:
            # Maximize blank entropy on contextual logits
            h_blank = blank_entropy(logits_g, blank=self.blank)
            # Negative sign because we maximising entropy ≡ minimising -H
            l_entropy = -h_blank
            losses["entropy"] = l_entropy.item()
            total = total + self.entropy_weight * l_entropy

        # ---- Optional: Self-distillation KD loss ----
        if self.self_distill and teacher_logits_g is not None:
            l_kd = self._kd_loss(logits_g, teacher_logits_g)
            losses["kd"] = l_kd.item()
            total = total + self.kd_weight * l_kd

        losses["total"] = total.item()
        return total, losses

    # ------------------------------------------------------------------
    def _weighted_ce_ls(self, logits, targets, weights):
        """Per-frame CE-LS with sample weights."""
        C = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)
        with torch.no_grad():
            smooth_targets = torch.full_like(log_probs, self.label_smoothing / (C - 1))
            valid_targets = targets.clone()
            smooth_targets.scatter_(-1, valid_targets.unsqueeze(-1), 1.0 - self.label_smoothing)
        loss = -(smooth_targets * log_probs).sum(dim=-1)
        loss = (loss * weights).sum() / weights.sum().clamp(min=1)
        return loss

    def _kd_loss(self, student_logits, teacher_logits):
        """KL-divergence KD loss with temperature scaling."""
        T = self.kd_temp
        s_log_probs = F.log_softmax(student_logits / T, dim=-1)
        t_probs = F.softmax(teacher_logits / T, dim=-1).detach()
        kl = F.kl_div(s_log_probs, t_probs, reduction="batchmean") * (T * T)
        return kl


# ---------------------------------------------------------------------------
# Focal CTC Loss – penalises hard examples more, soft on easy ones
# ---------------------------------------------------------------------------

class FocalCTCLoss(CTCLoss):
    """
    CTC loss with focal modulation: L_focal = (1 - p_target)^gamma * L_ctc
    where p_target is the posterior probability of the target alignment path.
    As an approximation, we use the per-frame max-prob along the Viterbi path.
    """
    def __init__(self, blank: int = 0, gamma: float = 2.0, zero_infinity: bool = True):
        super().__init__(blank=blank, zero_infinity=zero_infinity)
        self.gamma = gamma

    def forward(self, logits, targets, input_lengths, target_lengths):
        log_probs = F.log_softmax(logits, dim=-1)           # (B, T', C+1)
        log_probs_t = log_probs.permute(1, 0, 2)            # (T', B, C+1)
        probs_t = log_probs_t.exp()                          # (T', B, C+1)

        # Approximate focal weight: mean top-1 prob across frames
        with torch.no_grad():
            max_probs = probs_t.max(dim=-1).values.mean()    # scalar
            focal_weight = (1.0 - max_probs) ** self.gamma

        base_loss = self.ctc(log_probs_t, targets, input_lengths, target_lengths)
        return focal_weight * base_loss


# ---------------------------------------------------------------------------
# Entropy regularization helper (anti-spike: maximize blank entropy)
# ---------------------------------------------------------------------------

def blank_entropy(logits: torch.Tensor, blank: int = 0) -> torch.Tensor:
    """
    Compute entropy of the blank-class posterior.
    H = -p_blank * log(p_blank) - (1-p_blank) * log(1-p_blank)
    Returns mean entropy across all frames → maximise to discourage blank spikes.
    """
    probs = F.softmax(logits, dim=-1)                        # (..., C+1)
    p_blank = probs[..., blank]                              # (...)
    eps = 1e-8
    p_blank = p_blank.clamp(eps, 1.0 - eps)
    p_other = 1.0 - p_blank
    h = -(p_blank * torch.log(p_blank) + p_other * torch.log(p_other))
    return h.mean()


# ---------------------------------------------------------------------------
# Feature alignment loss (VAC-style): ||g_c - v_c|| between GCF and LVF
# ---------------------------------------------------------------------------

def feature_alignment_loss(
    lvf: torch.Tensor,           # (B, T', d)
    gcf: torch.Tensor,           # (B, T', d)
    mask: torch.Tensor,          # (B, T') bool – valid positions (e.g. GSBA anchor/expanded)
    mode: str = "cosine",
) -> torch.Tensor:
    """
    Compute distance between Local Visual Features and Global Contextual Features
    at masked positions.

    Args:
        lvf, gcf: feature tensors
        mask: (B, T') bool – True where alignment should be computed
        mode: "cosine" (1 - cosine_sim) or "l2" (euclidean distance)
    """
    if mask.sum() == 0:
        return lvf.new_tensor(0.0)

    lvf_m = lvf[mask]   # (N, d)
    gcf_m = gcf[mask]   # (N, d)

    if mode == "cosine":
        sim = F.cosine_similarity(lvf_m, gcf_m, dim=-1)      # (N,)
        return (1.0 - sim).mean()
    else:  # l2
        return F.mse_loss(lvf_m, gcf_m)  # MSE = ||diff||^2 / d


# ---------------------------------------------------------------------------
# GSBA – Gloss Segment Boundary Assignment  (with optional confidence margins)
# ---------------------------------------------------------------------------
