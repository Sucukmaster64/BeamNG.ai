# src/ml/datasets.py
# PyTorch Dataset + train/val split for BeamNG.ai segmentation
#
# Expects:
#   data/raw/*.png    (RGB images saved by OpenCV; usually BGR on disk)
#   data/labels/*.png (single-channel PNG with class ids 0..6)
#
# Usage:
#   from src.ml.datasets import make_dataloaders
#   train_loader, val_loader, meta = make_dataloaders(...)
#
# Notes:
# - Split is SEQUENCE/TIME-based (sorted filenames), not random per-frame.
# - For training, we apply lightweight augmentations (brightness/contrast, blur, noise, horizontal flip optional).
# - Labels are resized with NEAREST interpolation to preserve class ids.
#
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


# ---------- Config / Metadata ----------

@dataclass(frozen=True)
class SegDataMeta:
    root: str
    raw_dir: str
    label_dir: str
    num_classes: int
    ignore_index: int  # set to 0 if you want to ignore OTHER, else -100 for none


# ---------- Helpers ----------

def _list_pngs(folder: Path) -> List[Path]:
    return sorted([p for p in folder.glob("*.png") if p.is_file()])


def _stem(p: Path) -> str:
    return p.stem


def _read_bgr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img  # BGR


def _read_label(path: Path) -> np.ndarray:
    lbl = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if lbl is None:
        raise FileNotFoundError(f"Could not read label: {path}")
    # Ensure single channel uint8
    if lbl.ndim == 3:
        lbl = lbl[:, :, 0]
    if lbl.dtype != np.uint8:
        lbl = lbl.astype(np.uint8)
    return lbl


def _resize_image(img_bgr: np.ndarray, size_hw: Tuple[int, int]) -> np.ndarray:
    h, w = size_hw
    return cv2.resize(img_bgr, (w, h), interpolation=cv2.INTER_LINEAR)


def _resize_label(lbl: np.ndarray, size_hw: Tuple[int, int]) -> np.ndarray:
    h, w = size_hw
    return cv2.resize(lbl, (w, h), interpolation=cv2.INTER_NEAREST)


def _apply_augmentations(
    img_bgr: np.ndarray,
    lbl: np.ndarray,
    rng: np.random.Generator,
    hflip: bool = True,
    jitter: bool = True,
    blur: bool = True,
    noise: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    # Horizontal flip (safe for many road scenes; disable if you dislike it)
    if hflip and rng.random() < 0.5:
        img_bgr = cv2.flip(img_bgr, 1)
        lbl = cv2.flip(lbl, 1)

    if jitter:
        # brightness/contrast jitter in BGR space
        # contrast in [0.8, 1.2], brightness in [-20, 20]
        alpha = float(rng.uniform(0.8, 1.2))
        beta = float(rng.uniform(-20, 20))
        img_bgr = cv2.convertScaleAbs(img_bgr, alpha=alpha, beta=beta)

        # small gamma jitter
        if rng.random() < 0.5:
            gamma = float(rng.uniform(0.85, 1.15))
            inv = 1.0 / gamma
            table = (np.arange(256) / 255.0) ** inv * 255.0
            table = table.astype(np.uint8)
            img_bgr = cv2.LUT(img_bgr, table)

    if blur and rng.random() < 0.25:
        k = int(rng.choice([3, 5]))
        img_bgr = cv2.GaussianBlur(img_bgr, (k, k), 0)

    if noise and rng.random() < 0.25:
        sigma = float(rng.uniform(3.0, 10.0))
        n = rng.normal(0.0, sigma, size=img_bgr.shape).astype(np.float32)
        img_bgr = np.clip(img_bgr.astype(np.float32) + n, 0, 255).astype(np.uint8)

    return img_bgr, lbl


def _to_tensor(img_bgr: np.ndarray, lbl: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    # Convert BGR->RGB, HWC->CHW, uint8->float32 [0,1]
    img_rgb = img_bgr[:, :, ::-1].copy()
    x = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    y = torch.from_numpy(lbl.astype(np.int64))
    return x, y


# ---------- Dataset ----------

class BeamNGSegDataset(Dataset):
    def __init__(
        self,
        pairs: List[Tuple[Path, Path]],
        out_size_hw: Tuple[int, int] = (288, 512),
        training: bool = False,
        seed: int = 42,
        augment_hflip: bool = True,
    ):
        self.pairs = pairs
        self.out_size_hw = out_size_hw
        self.training = training
        self.augment_hflip = augment_hflip

        # Deterministic per-worker randomness handled in __getitem__ via index-based seed
        self.base_seed = int(seed)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        raw_path, lbl_path = self.pairs[idx]

        img = _read_bgr(raw_path)
        lbl = _read_label(lbl_path)

        # Resize to model input size
        img = _resize_image(img, self.out_size_hw)
        lbl = _resize_label(lbl, self.out_size_hw)

        if self.training:
            rng = np.random.default_rng(self.base_seed + idx)
            img, lbl = _apply_augmentations(
                img, lbl, rng,
                hflip=self.augment_hflip,
                jitter=True,
                blur=True,
                noise=True,
            )

        x, y = _to_tensor(img, lbl)

        return {
            "image": x,     # float32 [C,H,W] in RGB, range [0,1]
            "label": y,     # int64 [H,W], values 0..6
        }


# ---------- Pairing + Splits ----------

def build_pairs(
    root: str = "data",
    raw_subdir: str = "raw",
    label_subdir: str = "labels",
) -> List[Tuple[Path, Path]]:
    rootp = Path(root)
    raw_dir = rootp / raw_subdir
    lbl_dir = rootp / label_subdir

    raws = _list_pngs(raw_dir)
    lbls = _list_pngs(lbl_dir)

    if not raws:
        raise FileNotFoundError(f"No raw images found in: {raw_dir}")
    if not lbls:
        raise FileNotFoundError(f"No label images found in: {lbl_dir}")

    lbl_map = { _stem(p): p for p in lbls }
    pairs: List[Tuple[Path, Path]] = []

    missing = 0
    for r in raws:
        s = _stem(r)
        lp = lbl_map.get(s)
        if lp is None:
            missing += 1
            continue
        pairs.append((r, lp))

    if not pairs:
        raise RuntimeError("No matching raw/label pairs found (check filenames).")
    if missing > 0:
        print(f"[WARN] {missing} raw files had no matching label (ignored).")

    # Sequence/time-based ordering
    pairs.sort(key=lambda x: _stem(x[0]))
    return pairs


def split_pairs_sequential(
    pairs: List[Tuple[Path, Path]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.2,
) -> Tuple[List[Tuple[Path, Path]], List[Tuple[Path, Path]]]:
    if train_ratio <= 0 or val_ratio <= 0:
        raise ValueError("train_ratio and val_ratio must be > 0")
    if abs((train_ratio + val_ratio) - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio must equal 1.0")

    n = len(pairs)
    n_train = int(n * train_ratio)
    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:]
    return train_pairs, val_pairs


# ---------- Dataloaders ----------

def make_dataloaders(
    data_root: str = "data",
    raw_subdir: str = "raw",
    label_subdir: str = "labels",
    out_size_hw: Tuple[int, int] = (288, 512),
    batch_size: int = 16,
    num_workers: int = 4,
    seed: int = 42,
    train_ratio: float = 0.8,
    ignore_index: int = 0,  # set to 0 to ignore OTHER; set to -100 to not ignore anything
    augment_hflip: bool = True,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, SegDataMeta]:
    pairs = build_pairs(data_root, raw_subdir, label_subdir)
    train_pairs, val_pairs = split_pairs_sequential(pairs, train_ratio=train_ratio, val_ratio=1.0 - train_ratio)

    train_ds = BeamNGSegDataset(
        train_pairs,
        out_size_hw=out_size_hw,
        training=True,
        seed=seed,
        augment_hflip=augment_hflip,
    )
    val_ds = BeamNGSegDataset(
        val_pairs,
        out_size_hw=out_size_hw,
        training=False,
        seed=seed,
        augment_hflip=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,          # shuffle within train set is fine
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(1, num_workers // 2),
        pin_memory=pin_memory,
        drop_last=False,
    )

    meta = SegDataMeta(
        root=data_root,
        raw_dir=str(Path(data_root) / raw_subdir),
        label_dir=str(Path(data_root) / label_subdir),
        num_classes=7,
        ignore_index=ignore_index,
    )

    print(f"[INFO] Pairs total: {len(pairs)} | train: {len(train_pairs)} | val: {len(val_pairs)}")
    print(f"[INFO] Output size (H,W): {out_size_hw} | batch_size: {batch_size}")
    return train_loader, val_loader, meta
