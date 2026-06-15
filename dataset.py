
"""
Dataset utilities for a PHOENIX14 / PHOENIX14-T subset extracted with the
following flat layout:

  <data_root>/
      <annotation_dir>/                       e.g. "phoenix2014-T-pcet"
          PHOENIX-2014-T.train.corpus.csv
          PHOENIX-2014-T.dev.corpus.csv
          PHOENIX-2014-T.test.corpus.csv
      train/
          <sample_name>/                      e.g. "01December_2011_Thursday_heute-3060"
              images0001.png
              images0002.png
              ...
      dev/
          <sample_name>/...
      test/
          <sample_name>/...

Notes:
  - Each sample folder lives DIRECTLY under train/dev/test (no extra
    "features/fullFrame-210x260px/<split>/" nesting like the original
    PHOENIX14 release).
  - CSV files are pipe ('|') separated, with columns:
        name|video|start|end|speaker|orth|translation
    The 'orth' column holds the space-separated gloss sequence.
  - Image filenames are auto-detected (images*.png / *.png / *.jpg).
"""

import os
import csv
import glob
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

class Vocabulary:
    """Maps gloss strings <-> integer ids.  Index 0 is reserved for CTC blank."""

    BLANK = 0
    UNK = 1

    def __init__(self):
        self._w2i: Dict[str, int] = {"<blank>": 0, "<unk>": 1}
        self._i2w: Dict[int, str] = {0: "<blank>", 1: "<unk>"}

    def add(self, word: str) -> int:
        if word not in self._w2i:
            idx = len(self._w2i)
            self._w2i[word] = idx
            self._i2w[idx] = word
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
    """Full transform pipeline (used when no cache)."""
    base = [transforms.Resize(256)]
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
                              std=[0.229, 0.224, 0.225]),
    ])


def build_cache_transform():
    """
    Pre-cache transform: resize to 256, convert to tensor, no crop/normalize.
    Crop/flip/normalize are applied on-the-fly when loading from cache.
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])


def build_load_transform(split: str, img_size: int = 224):
    """
    On-the-fly transform applied when loading from cache.
    Assumes input is a (3, 256, 256) tensor.
    """
    if split == "train":
        aug = [
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
        ]
    else:
        aug = [transforms.CenterCrop(img_size)]

    return transforms.Compose(aug + [
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
    ])


# ---------------------------------------------------------------------------
# CSV discovery
# ---------------------------------------------------------------------------

def _find_corpus_csv(data_root: str, split: str) -> str:
    """
    Locates the corpus CSV for the given split.

    Search order:
      1. <data_root>/<*>/PHOENIX-2014-T.<split>.corpus.csv
      2. <data_root>/PHOENIX-2014-T.<split>.corpus.csv
      3. <data_root>/<*>/*<split>*.corpus.csv  (fallback, looser match)
    """
    patterns = [
        os.path.join(data_root, "*", f"PHOENIX-2014-T.{split}.corpus.csv"),
        os.path.join(data_root, f"PHOENIX-2014-T.{split}.corpus.csv"),
        os.path.join(data_root, "*", f"*.{split}.corpus.csv"),
        os.path.join(data_root, f"*.{split}.corpus.csv"),
        os.path.join(data_root, "**", f"*{split}*corpus*.csv"),
    ]
    for pattern in patterns:
        matches = sorted(glob.glob(pattern, recursive=True))
        if matches:
            return matches[0]

    raise FileNotFoundError(
        f"Could not find a corpus CSV for split='{split}' under '{data_root}'. "
        f"Expected something like '<data_root>/<annotation_dir>/"
        f"PHOENIX-2014-T.{split}.corpus.csv'."
    )


# ---------------------------------------------------------------------------
# PHOENIX-style Dataset (flat per-sample folders)
# ---------------------------------------------------------------------------

class PhoenixDataset(Dataset):
    """
    Loads pre-extracted frames from a flat PHOENIX14/14-T subset:

        <data_root>/<split>/<sample_name>/<frame images>

    where <sample_name> matches the 'name' (or 'video' folder stem) column
    of the corpus CSV.
    """

    # File extensions to look for when listing frames, in priority order.
    FRAME_GLOBS = ("*.png", "*.jpg", "*.jpeg")

    def __init__(
        self,
        root: str,
        split: str,
        vocab: Optional[Vocabulary] = None,
        img_size: int = 224,
        max_frames: int = 300,
        synthetic: bool = False,
        # Frame caching
        cache_dir: Optional[str] = None,
        # Temporal augmentation params
        temporal_aug_enabled: bool = False,
        temporal_mask_prob: float = 0.3,
        temporal_mask_max_ratio: float = 0.15,
        temporal_jitter_range: float = 0.0,
    ):
        assert split in ("train", "dev", "test")
        self.root = root
        self.split = split
        self.img_size = img_size
        self.max_frames = max_frames
        self.synthetic = synthetic
        self.transform = build_transforms(split, img_size)  # fallback (no cache)

        # Frame caching
        self.cache_dir = cache_dir
        self.cache_transform = build_cache_transform()
        self.load_transform = build_load_transform(split, img_size)
        self.cache_hits = 0
        self.cache_misses = 0

        # Temporal augmentation (only applied during training)
        self.temporal_aug_enabled = temporal_aug_enabled and split == "train"
        self.temporal_mask_prob = temporal_mask_prob
        self.temporal_mask_max_ratio = temporal_mask_max_ratio
        self.temporal_jitter_range = temporal_jitter_range

        if not synthetic:
            self.samples = self._load_annotations()
        else:
            self.samples = self._make_synthetic_samples()

        if vocab is None:
            self.vocab = Vocabulary.build(self.samples)
        else:
            self.vocab = vocab

    # ------------------------------------------------------------------
    def _load_annotations(self) -> List[dict]:
        csv_path = _find_corpus_csv(self.root, self.split)
        split_dir = os.path.join(self.root, self.split)

        samples = []
        skipped = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="|")
            for row in reader:
                # 'name' is the canonical sample id; fall back to the stem
                # of 'video' (e.g. ".../01December_2011.../1/*.png" -> folder name)
                name = row.get("name", "").strip()
                if not name:
                    video = row.get("video", "")
                    name = video.split("/")[0] if video else ""
                if not name:
                    continue

                orth = row.get("orth", "")
                glosses = orth.strip().split() if orth else []

                sample_dir = os.path.join(split_dir, name)
                if not os.path.isdir(sample_dir):
                    skipped.append(name)
                    continue

                samples.append({
                    "id": name,
                    "frame_dir": sample_dir,
                    "glosses": glosses,
                })

        if not samples:
            raise RuntimeError(
                f"No samples found for split='{self.split}'. "
                f"Checked CSV '{csv_path}' against folder '{split_dir}'. "
                f"{len(skipped)} CSV rows had no matching folder."
            )
        if skipped:
            print(
                f"[PhoenixDataset] split='{self.split}': "
                f"{len(skipped)} CSV entries had no matching folder under "
                f"'{split_dir}' and were skipped (e.g. {skipped[:3]})."
            )

        return samples

    def _make_synthetic_samples(self, n: int = 32) -> List[dict]:
        glosses_pool = [f"GLOSS_{i}" for i in range(20)]
        samples = []
        for i in range(n):
            seq_len = torch.randint(4, 9, (1,)).item()
            glosses = [glosses_pool[j % 20] for j in range(seq_len)]
            samples.append({
                "id": f"synthetic_{i}",
                "frame_dir": None,
                "glosses": glosses,
                "n_frames": torch.randint(40, 100, (1,)).item(),
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
            frames = torch.randn(n_frames, 3, self.img_size, self.img_size)
        elif self.cache_dir is not None:
            # Try to load pre-cached frames (resized tensor), then apply crop/flip/norm
            frames = self._load_cached_or_process(sample)
        else:
            frames = self._load_frames(sample["frame_dir"])

        # Apply temporal augmentations (training only)
        if self.temporal_aug_enabled:
            frames = self._apply_temporal_aug(frames)

        label_ids = torch.tensor(
            self.vocab.encode(sample["glosses"]), dtype=torch.long
        )
        return {
            "frames": frames,        # (T, 3, H, W)
            "labels": label_ids,     # (N_gloss,)
            "gloss_ids": self.vocab.encode(sample["glosses"]),
            "id": sample["id"],
        }

    def _cache_path(self, sample_id: str) -> str:
        """Path to the cached frame tensor for a sample."""
        # Include img_size in key since it affects the subsampled frame count
        key = f"{sample_id}_sz{self.img_size}_mf{self.max_frames}.pt"
        return os.path.join(self.cache_dir, key)

    def _load_cached_or_process(self, sample: dict) -> torch.Tensor:
        """
        Load frames from cache if available; otherwise process and save to cache.

        Cache stores (T, 3, 256, 256) tensors (resized, not cropped/normalized).
        On load, apply RandomCrop/CenterCrop + Normalize on-the-fly.
        """
        cache_path = self._cache_path(sample["id"])

        if os.path.exists(cache_path):
            self.cache_hits += 1
            frames = torch.load(cache_path, map_location="cpu", weights_only=True)
            # Apply crop + normalize on-the-fly
            frames = torch.stack([self.load_transform(f) for f in frames])
            return frames

        # Cache miss — process frames and save
        self.cache_misses += 1
        raw_frames = self._load_frames_raw(sample["frame_dir"])
        # Save resized-but-not-cropped tensors to cache
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(raw_frames.cpu(), cache_path)
        # Apply crop + normalize
        frames = torch.stack([self.load_transform(f) for f in raw_frames])
        return frames

    def cache_stats(self) -> dict:
        """Return cache hit/miss statistics."""
        return {
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "total": self.cache_hits + self.cache_misses,
            "hit_rate": self.cache_hits / max(1, self.cache_hits + self.cache_misses),
        }

    def _apply_temporal_aug(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Apply temporal augmentations to a frame sequence.
        - Temporal masking: zero out a random contiguous block of frames
        - Temporal jitter: speed up/slow down via random frame rate change
        """
        T = frames.size(0)
        if T < 2:
            return frames

        # 1. Temporal masking
        if self.temporal_mask_prob > 0 and torch.rand(1).item() < self.temporal_mask_prob:
            mask_len = max(1, int(T * self.temporal_mask_max_ratio * torch.rand(1).item()))
            mask_start = torch.randint(0, max(1, T - mask_len), (1,)).item()
            frames = frames.clone()
            frames[mask_start:mask_start + mask_len] = 0.0

        # 2. Temporal jitter (rate variation)
        if self.temporal_jitter_range > 0:
            jitter = 1.0 + self.temporal_jitter_range * (2 * torch.rand(1).item() - 1)
            new_T = max(2, min(self.max_frames, int(T * jitter)))
            if new_T != T:
                # Simple nearest-neighbour temporal resampling
                indices = torch.linspace(0, T - 1, new_T).long()
                frames = frames[indices]

        return frames

    def _load_frames_raw(self, frame_dir: str) -> torch.Tensor:
        """
        Load frames, apply ONLY Resize+ToTensor (no crop, no normalize).
        Returns: (T, 3, 256, 256) tensor — ready for caching.
        """
        paths: List[str] = []
        for pattern in self.FRAME_GLOBS:
            paths = sorted(glob.glob(os.path.join(frame_dir, pattern)))
            if paths:
                break

        if not paths:
            raise FileNotFoundError(
                f"No image frames found in '{frame_dir}' "
                f"(looked for {self.FRAME_GLOBS})."
            )

        # Sub-sample uniformly to max_frames
        if len(paths) > self.max_frames:
            step = len(paths) / self.max_frames
            paths = [paths[int(i * step)] for i in range(self.max_frames)]

        frames = [self.cache_transform(Image.open(p).convert("RGB")) for p in paths]
        return torch.stack(frames)  # (T, 3, 256, 256)

    def _load_frames(self, frame_dir: str) -> torch.Tensor:
        """
        Full pipeline: load images, apply all transforms (resize, crop, normalize).
        Used when cache_dir is None.
        Returns: (T, 3, img_size, img_size) normalized tensor.
        """
        paths: List[str] = []
        for pattern in self.FRAME_GLOBS:
            paths = sorted(glob.glob(os.path.join(frame_dir, pattern)))
            if paths:
                break

        if not paths:
            raise FileNotFoundError(
                f"No image frames found in '{frame_dir}' "
                f"(looked for {self.FRAME_GLOBS})."
            )

        # Sub-sample uniformly to max_frames
        if len(paths) > self.max_frames:
            step = len(paths) / self.max_frames
            paths = [paths[int(i * step)] for i in range(self.max_frames)]

        frames = [self.transform(Image.open(p).convert("RGB")) for p in paths]
        return torch.stack(frames)  # (T, 3, H, W)


# ---------------------------------------------------------------------------
# Collate function (handles variable-length sequences)
# ---------------------------------------------------------------------------

def collate_fn(batch: List[dict]):
    """Pad frames to the same T within a batch."""
    frames_list = [b["frames"] for b in batch]
    labels_list = [b["labels"] for b in batch]
    ids = [b["id"] for b in batch]
    gloss_seqs = [b["gloss_ids"] for b in batch]

    T_max = max(f.size(0) for f in frames_list)
    B = len(batch)
    C, H, W = frames_list[0].shape[1:]

    padded_frames = torch.zeros(B, T_max, C, H, W)
    input_lengths = torch.zeros(B, dtype=torch.long)

    for i, f in enumerate(frames_list):
        T = f.size(0)
        padded_frames[i, :T] = f
        input_lengths[i] = T

    flat_labels = torch.cat(labels_list) if labels_list else torch.empty(0, dtype=torch.long)
    target_lengths = torch.tensor([len(l) for l in labels_list], dtype=torch.long)

    return {
        "frames": padded_frames,         # (B, T_max, C, H, W)
        "labels": flat_labels,           # (sum_N,)
        "input_lengths": input_lengths,  # (B,)
        "target_lengths": target_lengths,  # (B,)
        "ids": ids,
        "gloss_seqs": gloss_seqs,         # list of B lists
    }