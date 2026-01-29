# src/ml/train_seg.py
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Tuple

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from src.ml.datasets import make_dataloaders
from src.ml.models.enet import ENet


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def confusion_matrix(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: int = 0,
) -> torch.Tensor:
    """
    pred: (N,H,W) int64
    target: (N,H,W) int64
    returns: (C,C) where rows=true, cols=pred
    """
    if ignore_index is not None and ignore_index >= 0:
        mask = target != ignore_index
        pred = pred[mask]
        target = target[mask]

    k = (target >= 0) & (target < num_classes)
    pred = pred[k]
    target = target[k]
    idx = target * num_classes + pred
    cm = torch.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    return cm


@torch.no_grad()
def compute_iou(cm: torch.Tensor) -> Tuple[float, Dict[int, float]]:
    """
    cm: (C,C)
    returns: (mean_iou, per_class_iou)
    """
    cm = cm.float()
    diag = torch.diag(cm)
    denom = cm.sum(1) + cm.sum(0) - diag
    iou = torch.where(denom > 0, diag / denom, torch.zeros_like(denom))
    miou = iou.mean().item()
    per = {int(i): float(iou[i].item()) for i in range(cm.shape[0])}
    return miou, per


def default_class_weights(num_classes: int = 7) -> torch.Tensor:
    # 0 OTHER -> very low or ignored via ignore_index
    # 1 ROAD
    # 2 SHOULDER
    # 3 SIDEWALK
    # 4 TERRAIN
    # 5 OBSTACLE
    # 6 MARKING
    w = torch.ones(num_classes, dtype=torch.float32)
    w[0] = 0.05
    w[1] = 1.0
    w[2] = 2.0
    w[3] = 2.0
    w[4] = 1.2
    w[5] = 1.2
    w[6] = 3.0
    return w


def build_model(name: str, num_classes: int) -> nn.Module:
    name = name.lower().strip()
    if name == "enet":
        return ENet(num_classes=num_classes)
    raise ValueError(f"Unknown model: {name}")


def save_checkpoint(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, str(path))


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer,
    scaler: GradScaler,
    criterion: nn.Module,
    device: torch.device,
    scheduler=None,
    amp: bool = True,
    epoch: int = 1,
    log_every: int = 50,
    writer: SummaryWriter | None = None,
) -> float:
    model.train()
    running = 0.0
    n = 0

    pbar = tqdm(loader, desc=f"train e{epoch}", ncols=110, leave=False)
    for step, batch in enumerate(pbar, start=1):
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=amp):
            logits = model(x)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        bs = x.size(0)
        running += float(loss.item()) * bs
        n += bs

        avg = running / max(1, n)
        lr = optimizer.param_groups[0]["lr"]
        pbar.set_postfix(loss=f"{avg:.4f}", lr=f"{lr:.2e}")

        # optional per-step logging to TensorBoard
        if writer is not None and (step % log_every == 0):
            global_step = (epoch - 1) * len(loader) + step
            writer.add_scalar("loss/train_step", float(loss.item()), global_step)
            writer.add_scalar("lr/step", lr, global_step)

    return running / max(1, n)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    ignore_index: int,
    amp: bool = True,
    epoch: int = 1,
) -> Tuple[float, float, Dict[int, float]]:
    model.eval()
    running = 0.0
    n = 0

    cm_total = torch.zeros((num_classes, num_classes), dtype=torch.int64, device="cpu")

    pbar = tqdm(loader, desc=f"val   e{epoch}", ncols=110, leave=False)
    for batch in pbar:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)

        with autocast(enabled=amp):
            logits = model(x)
            loss = criterion(logits, y)

        bs = x.size(0)
        running += float(loss.item()) * bs
        n += bs

        pred = torch.argmax(logits, dim=1).to(torch.int64)
        cm = confusion_matrix(pred.cpu(), y.cpu(), num_classes=num_classes, ignore_index=ignore_index)
        cm_total += cm

        avg = running / max(1, n)
        pbar.set_postfix(loss=f"{avg:.4f}")

    val_loss = running / max(1, n)
    miou, per_class = compute_iou(cm_total)
    return val_loss, miou, per_class


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="enet", choices=["enet"])
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--img-size", type=int, nargs=2, default=[288, 512], metavar=("H", "W"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--ignore-index", type=int, default=0)  # 0 = OTHER
    parser.add_argument("--no-hflip", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = (not args.no_amp) and device.type == "cuda"

    # Speed knobs (safe defaults)
    torch.backends.cudnn.benchmark = True

    train_loader, val_loader, meta = make_dataloaders(
        data_root=args.data_root,
        out_size_hw=(args.img_size[0], args.img_size[1]),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_ratio=args.train_ratio,
        ignore_index=args.ignore_index,
        augment_hflip=(not args.no_hflip),
    )

    num_classes = meta.num_classes

    model = build_model(args.model, num_classes=num_classes).to(device)

    weights = default_class_weights(num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=args.ignore_index)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # OneCycleLR gives fast convergence for small models
    steps_per_epoch = len(train_loader)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,
        div_factor=10.0,
        final_div_factor=100.0,
    )

    scaler = GradScaler(enabled=amp)

    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path("runs") / f"{args.model}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(run_dir))


    # Save config
    cfg = {
        "args": vars(args),
        "meta": asdict(meta),
        "class_weights": weights.detach().cpu().tolist(),
        "device": str(device),
        "amp": bool(amp),
    }
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    print(f"[INFO] device={device} amp={amp}")
    print(f"[INFO] run_dir={run_dir}")

    best_miou = -1.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss = train_one_epoch(model, train_loader, optimizer, scaler, criterion, device, scheduler=scheduler, amp=amp, epoch=epoch, log_every=50, writer=writer)
        val_loss, miou, per = validate(model, val_loader, criterion, device, num_classes, args.ignore_index, amp=amp, epoch=epoch)
        dt = time.time() - t0

        # Log
        msg = (
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={tr_loss:.4f} val_loss={val_loss:.4f} miou={miou:.4f} | "
            f"time={dt:.1f}s"
        )
        print(msg)

        # Scalars
        writer.add_scalar("loss/train", tr_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("metrics/miou", miou, epoch)

        # LR (OneCycle ändert LR pro Step; wir loggen den aktuellen)
        current_lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("lr", current_lr, epoch)

        # Per-class IoU
        for cls_id, v in per.items():
            writer.add_scalar(f"iou/class_{cls_id}", v, epoch)

        writer.flush()


        # Save last
        save_checkpoint(run_dir / "last.pt", {
            "epoch": epoch,
            "model": args.model,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "miou": miou,
            "val_loss": val_loss,
            "args": vars(args),
        })

        # Save best
        if miou > best_miou:
            best_miou = miou
            save_checkpoint(run_dir / "best.pt", {
                "epoch": epoch,
                "model": args.model,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "miou": miou,
                "val_loss": val_loss,
                "args": vars(args),
                "per_class_iou": per,
            })
            (run_dir / "best_metrics.json").write_text(json.dumps({
                "epoch": epoch,
                "miou": miou,
                "val_loss": val_loss,
                "per_class_iou": per,
            }, indent=2), encoding="utf-8")
            print(f"[INFO] New best mIoU: {best_miou:.4f}")

    print(f"[DONE] Best mIoU: {best_miou:.4f}")
    print(f"[DONE] Artifacts in: {run_dir}")
    writer.close()


if __name__ == "__main__":
    main()
