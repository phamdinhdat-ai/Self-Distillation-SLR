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
) -> torch.Tensor:
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
    """
    T = gcf.size(0)
    seg_labels = torch.full((T,), -1, dtype=torch.long, device=gcf.device)

    # Normalise features and weights for cosine sim
    gcf_n = F.normalize(gcf, p=2, dim=-1)                  # (T', d)
    w_n   = F.normalize(classifier_weights, p=2, dim=-1)   # (C+1, d)

    # Only cosine sims for classes actually present in this sentence
    gloss_ids = torch.tensor(gloss_seq, device=gcf.device)  # (N_gloss,)
    w_gloss   = w_n[gloss_ids]                              # (N_gloss, d)

    def best_class(t_idx: int) -> int:
        sims = gcf_n[t_idx] @ w_gloss.t()   # (N_gloss,)
        best = int(sims.argmax().item())
        return int(gloss_ids[best].item())

    # Anchor frames
    anchor_mask = (spike_labels != blank)
    anchor_indices = anchor_mask.nonzero(as_tuple=True)[0].tolist()

    for t in anchor_indices:
        c_a = int(spike_labels[t].item())
        seg_labels[t] = c_a

        # Expand backwards
        for dt in range(1, d + 1):
            ti = t - dt
            if ti < 0:
                break
            if seg_labels[ti] != -1:
                break
            if best_class(ti) == c_a:
                seg_labels[ti] = c_a
            else:
                break

        # Expand forwards
        for dt in range(1, d + 1):
            ti = t + dt
            if ti >= T:
                break
            if seg_labels[ti] != -1:
                break
            if best_class(ti) == c_a:
                seg_labels[ti] = c_a
            else:
                break

    return seg_labels


def batch_gsba(
    gcf: torch.Tensor,                  # (B, T', d)
    classifier_weights: torch.Tensor,   # (C+1, d) – already normalised
    spike_labels: torch.Tensor,         # (B, T')
    gloss_seqs: List[List[int]],        # list of B gloss sequences
    d: int = 1,
    blank: int = 0,
) -> torch.Tensor:
    """Run GSBA for a full batch; returns (B, T') segment labels."""
    B, T = spike_labels.shape
    seg = torch.full((B, T), -1, dtype=torch.long, device=gcf.device)
    for b in range(B):
        seg[b] = gsba(
            gcf[b], classifier_weights,
            spike_labels[b], gloss_seqs[b],
            d=d, blank=blank,
        )
    return seg


# ---------------------------------------------------------------------------
# SMKD Training Loss (all stages)
# ---------------------------------------------------------------------------

class SMKDLoss(nn.Module):
    """
    Combines the per-stage training objectives:

    Stage 1 (sync train):
        L = CTC(g) + α * CTC(v)

    Stage 2 (gloss segmentation):
        L = CTC(g) + α * CE-LS(v, seg_labels)

    Stage 3 (decouple train):
        L = CTC(g)   [visual classifier detached; contextual only]
    """

    def __init__(
        self,
        blank: int = 0,
        alpha: float = 0.5,
        label_smoothing: float = 0.2,
    ):
        super().__init__()
        self.ctc_loss = CTCLoss(blank=blank)
        self.ce_ls    = LabelSmoothingCE(smoothing=label_smoothing, ignore_index=-1)
        self.alpha    = alpha
        self.blank    = blank

    def forward(
        self,
        logits_v: torch.Tensor,          # (B, T', C+1)
        logits_g: torch.Tensor,          # (B, T', C+1)
        targets: torch.Tensor,           # (sum targets,) flat
        input_lengths: torch.Tensor,     # (B,)
        target_lengths: torch.Tensor,    # (B,)
        stage: int,
        seg_labels: torch.Tensor = None, # (B, T')  required for stage 2
    ) -> Tuple[torch.Tensor, dict]:

        losses = {}

        # CTC on contextual features (all stages)
        l_ctc_g = self.ctc_loss(logits_g, targets, input_lengths, target_lengths)
        losses["ctc_g"] = l_ctc_g.item()

        if stage == 1:
            l_ctc_v = self.ctc_loss(logits_v, targets, input_lengths, target_lengths)
            losses["ctc_v"] = l_ctc_v.item()
            total = l_ctc_g + self.alpha * l_ctc_v

        elif stage == 2:
            # CE-LS on visual features with GSBA segment labels
            assert seg_labels is not None, "seg_labels required for stage 2"
            B, T_prime, C1 = logits_v.shape
            # Flatten (B, T') → (B*T',)
            logits_v_flat   = logits_v.view(-1, C1)
            seg_labels_flat = seg_labels.view(-1)
            l_seg = self.ce_ls(logits_v_flat, seg_labels_flat)
            losses["seg"] = l_seg.item()
            total = l_ctc_g + self.alpha * l_seg

        else:  # stage 3 – decouple
            total = l_ctc_g

        losses["total"] = total.item()
        return total, losses
