"""
Dataset utilities for PHOENIX14 / PHOENIX14-T style CSLR datasets.

Expected directory layout (PHOENIX14):
  <root>/
    features/fullFrame-210x260px/
      train/<signer>/<sample_id>/*.png
      dev/   ...
      test/  ...
    annotations/manual/
      train.corpus.csv
      dev.corpus.csv
      test.corpus.csv

CSV columns (pipe-separated):
  name|video|start|end|speaker|orth|translation

'orth' contains the space-separated gloss sequence.
"""

import os
import csv
import glob
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

class Vocabulary:
    """Maps gloss strings ↔ integer ids.  Index 0 is reserved for CTC blank."""

    BLANK = 0
    UNK   = 1

    def __init__(self):
        self._w2i: Dict[str, int] = {"<blank>": 0, "<unk>": 1}
        self._i2w: Dict[int, str] = {0: "<blank>", 1: "<unk>"}

    def add(self, word: str) -> int:
        if word not in self._w2i:
            idx = len(self._w2i)
            self._w2i[word] = idx
            self._i2w[idx]  = word
        return self._w2i[word]

    def __len__(self):
        return len(self._w2i)

    def encode(self, gloss_seq: List[str]) -> List[int]:
        return [self._w2i.get(g, self.UNK) for g in gloss_seq]

    def decode(self, ids: List[int]) -> List[str]:
        return [self._i2w.get(i, "<unk>") for i in ids]

    @classmethod
    def build(cls, samples: List[dict]) -> "Vocabulary":
        vocab = cls()
        for s in samples:
            for g in s["glosses"]:
                vocab.add(g)
        return vocab


# ---------------------------------------------------------------------------
# Frame transforms
# ---------------------------------------------------------------------------

def build_transforms(split: str, img_size: int = 224):
    base = [
        transforms.Resize(256),
    ]
    if split == "train":
        aug = [
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
        ]
    else:
        aug = [transforms.CenterCrop(img_size)]

    return transforms.Compose(base + aug + [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])


# ---------------------------------------------------------------------------
# Phoenix-style Dataset
# ---------------------------------------------------------------------------

class PhoenixDataset(Dataset):
    """
    Loads pre-extracted frames from a PHOENIX14-style dataset.
    Falls back to synthetic random tensors when frames are not found
    (useful for unit-testing without the actual corpus).
    """

    def __init__(
        self,
        root: str,
        split: str,
        vocab: Optional[Vocabulary] = None,
        img_size: int = 224,
        max_frames: int = 300,
        synthetic: bool = False,
    ):
        assert split in ("train", "dev", "test")
        self.root       = root
        self.split      = split
        self.img_size   = img_size
        self.max_frames = max_frames
        self.synthetic  = synthetic
        self.transform  = build_transforms(split, img_size)

        # Load annotations
        if not synthetic:
            self.samples = self._load_annotations()
        else:
            # Create tiny synthetic dataset for testing
            self.samples = self._make_synthetic_samples()

        # Build / use vocabulary
        if vocab is None:
            self.vocab = Vocabulary.build(self.samples)
        else:
            self.vocab = vocab

    # ------------------------------------------------------------------
    def _load_annotations(self) -> List[dict]:
        csv_path = os.path.join(
            self.root, "annotations", "manual", f"{self.split}.corpus.csv"
        )
        samples = []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f, delimiter="|")
            for row in reader:
                name    = row["name"]
                glosses = row["orth"].strip().split()
                frame_dir = os.path.join(
                    self.root,
                    "features", "fullFrame-210x260px",
                    self.split, name,
                )
                samples.append({
                    "id":       name,
                    "frame_dir": frame_dir,
                    "glosses":  glosses,
                })
        return samples

    def _make_synthetic_samples(self, n: int = 32) -> List[dict]:
        glosses_pool = [f"GLOSS_{i}" for i in range(20)]
        samples = []
        for i in range(n):
            seq_len  = torch.randint(4, 9, (1,)).item()
            glosses  = [glosses_pool[j % 20] for j in range(seq_len)]
            samples.append({
                "id":        f"synthetic_{i}",
                "frame_dir": None,
                "glosses":   glosses,
                "n_frames":  torch.randint(40, 100, (1,)).item(),
            })
        return samples

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]

        if self.synthetic or sample["frame_dir"] is None:
            n_frames = int(sample.get("n_frames", 60))
            n_frames = min(n_frames, self.max_frames)
            # Random RGB frames  (T, C, H, W)
            frames = torch.randn(n_frames, 3, self.img_size, self.img_size)
        else:
            frames = self._load_frames(sample["frame_dir"])

        label_ids = torch.tensor(
            self.vocab.encode(sample["glosses"]), dtype=torch.long
        )
        return {
            "frames":      frames,          # (T, 3, H, W)
            "labels":      label_ids,       # (N_gloss,)
            "gloss_ids":   self.vocab.encode(sample["glosses"]),
            "id":          sample["id"],
        }

    def _load_frames(self, frame_dir: str) -> torch.Tensor:
        paths = sorted(glob.glob(os.path.join(frame_dir, "*.png")))
        if not paths:
            paths = sorted(glob.glob(os.path.join(frame_dir, "*.jpg")))
        # Sub-sample to max_frames
        if len(paths) > self.max_frames:
            step = len(paths) / self.max_frames
            paths = [paths[int(i * step)] for i in range(self.max_frames)]
        frames = []
        for p in paths:
            img = Image.open(p).convert("RGB")
            frames.append(self.transform(img))
        return torch.stack(frames)   # (T, 3, H, W)


# ---------------------------------------------------------------------------
# Collate function (handles variable-length sequences)
# ---------------------------------------------------------------------------

def collate_fn(batch: List[dict]):
    """Pad frames to the same T within a batch."""
    frames_list = [b["frames"] for b in batch]
    labels_list = [b["labels"] for b in batch]
    ids         = [b["id"] for b in batch]
    gloss_seqs  = [b["gloss_ids"] for b in batch]

    # Input lengths (T' after visual module ≈ T // 2, but we pass T for loss)
    T_max = max(f.size(0) for f in frames_list)
    B     = len(batch)
    C, H, W = frames_list[0].shape[1:]

    padded_frames  = torch.zeros(B, T_max, C, H, W)
    input_lengths  = torch.zeros(B, dtype=torch.long)

    for i, f in enumerate(frames_list):
        T = f.size(0)
        padded_frames[i, :T] = f
        input_lengths[i]     = T

    # Flatten labels
    flat_labels    = torch.cat(labels_list)
    target_lengths = torch.tensor([len(l) for l in labels_list], dtype=torch.long)

    return {
        "frames":         padded_frames,    # (B, T_max, C, H, W)
        "labels":         flat_labels,      # (sum_N,)
        "input_lengths":  input_lengths,    # (B,)
        "target_lengths": target_lengths,   # (B,)
        "ids":            ids,
        "gloss_seqs":     gloss_seqs,       # list of B lists
    }
