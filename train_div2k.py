"""
M23-RLFN Training Script on DIV2K
===================================

Полный тренировочный пайплайн:
- Датасет DIV2K (локальная папка)
- Аугментации: random crop, flip, rotation90
- AMP + gradient clipping
- Multi-stage warm-start LR
- PSNR валидация каждые N итераций
- Чекпоинты + resume
- Логирование в консоль и CSV

Запуск:
    python train_div2k.py --scale 4 --total_iters 300000 --batch_size 16

Структура папок (ожидается):
    data/
        DIV2K_train_HR/   *.png
        DIV2K_valid_HR/   *.png
    (LR генерируется on-the-fly через bicubic downsample)
"""

import os
import csv
import time
import argparse
import random
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF
from PIL import Image

from m23_sr_engine import M23_RLFN, SRTrainer, CombinedSRLoss, WarmStartScheduler


# ══════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════

class DIV2KDataset(Dataset):
    """
    DIV2K датасет с on-the-fly LR генерацией.
    HR → bicubic downsample → LR пара.
    Поддерживает случайный crop, flip, rotation90.
    """

    def __init__(
        self,
        hr_dir:     str,
        scale:      int  = 4,
        patch_size: int  = 256,
        augment:    bool = True,
        cache:      bool = True,
    ):
        self.hr_dir     = Path(hr_dir)
        self.scale      = scale
        self.patch_size = patch_size
        self.augment    = augment
        self.cache      = cache
        self.hr_paths: List[Path] = sorted(
            p for p in self.hr_dir.iterdir()
            if p.suffix.lower() in (".png", ".jpg", ".jpeg")
        )
        if not self.hr_paths:
            raise FileNotFoundError(f"No images found in {hr_dir}")
        self._cache: Dict[int, torch.Tensor] = {}
        print(f"[DIV2K] {len(self.hr_paths)} images | scale=×{scale} | "
              f"patch={patch_size}px | augment={augment} | cache={cache}")

    def _load_hr(self, idx: int) -> torch.Tensor:
        if self.cache and idx in self._cache:
            return self._cache[idx]
        t = TF.to_tensor(Image.open(self.hr_paths[idx]).convert("RGB"))
        if self.cache:
            self._cache[idx] = t
        return t

    def _random_crop(self, hr: torch.Tensor) -> torch.Tensor:
        _, H, W = hr.shape
        ps = self.patch_size
        if H < ps or W < ps:
            hr = TF.pad(hr, [0, 0, max(0, ps - W), max(0, ps - H)])
            _, H, W = hr.shape
        top  = random.randint(0, H - ps)
        left = random.randint(0, W - ps)
        return hr[:, top:top + ps, left:left + ps]

    def _augment(self, hr: torch.Tensor) -> torch.Tensor:
        if random.random() > 0.5:
            hr = TF.hflip(hr)
        if random.random() > 0.5:
            hr = TF.vflip(hr)
        k = random.randint(0, 3)
        if k > 0:
            hr = torch.rot90(hr, k, dims=[1, 2])
        return hr

    def __len__(self) -> int:
        return len(self.hr_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        hr      = self._load_hr(idx)
        hr      = self._random_crop(hr)
        if self.augment:
            hr  = self._augment(hr)
        lr_size = self.patch_size // self.scale
        lr = F.interpolate(
            hr.unsqueeze(0),
            size=(lr_size, lr_size),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        ).squeeze(0).clamp(0, 1)
        return lr, hr


class DIV2KValDataset(Dataset):
    """Валидационный датасет — полные изображения без аугментаций."""

    def __init__(self, hr_dir: str, scale: int = 4, max_size: int = 100):
        self.hr_paths = sorted(
            p for p in Path(hr_dir).iterdir()
            if p.suffix.lower() in (".png", ".jpg", ".jpeg")
        )[:max_size]
        self.scale = scale

    def __len__(self):
        return len(self.hr_paths)

    def __getitem__(self, idx):
        hr      = TF.to_tensor(Image.open(self.hr_paths[idx]).convert("RGB"))
        _, H, W = hr.shape
        H = (H // self.scale) * self.scale
        W = (W // self.scale) * self.scale
        hr = hr[:, :H, :W]
        lr = F.interpolate(
            hr.unsqueeze(0),
            size=(H // self.scale, W // self.scale),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        ).squeeze(0).clamp(0, 1)
        return lr, hr


# ══════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════

def calc_psnr(pred: torch.Tensor, target: torch.Tensor, border: int = 0) -> float:
    """PSNR на Y-канале (BT.601), как в NTIRE бенчмарках."""
    if border > 0:
        pred   = pred[...,   border:-border, border:-border]
        target = target[..., border:-border, border:-border]

    def rgb_to_y(t):
        r, g, b = t[:, 0:1], t[:, 1:2], t[:, 2:3]
        return 0.299 * r + 0.587 * g + 0.114 * b

    mse = F.mse_loss(rgb_to_y(pred.clamp(0, 1)), rgb_to_y(target.clamp(0, 1))).item()
    return 100.0 if mse < 1e-10 else -10 * np.log10(mse)


# ══════════════════════════════════════════════════════════
# CHECKPOINT
# ══════════════════════════════════════════════════════════

def save_checkpoint(
    path:      str,
    model:     nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmStartScheduler,
    iteration: int,
    best_psnr: float,
    scaler=None,
):
    ckpt = {
        "iteration": iteration,
        "best_psnr": best_psnr,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": {
            "current_iter":  scheduler.current_iter,
            "current_stage": scheduler.current_stage,
        },
    }
    if scaler is not None:
        ckpt["scaler"] = scaler.state_dict()
    torch.save(ckpt, path)


def load_checkpoint(
    path:      str,
    model:     nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmStartScheduler,
    scaler=None,
) -> Tuple[int, float]:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.current_iter  = ckpt["scheduler"]["current_iter"]
    scheduler.current_stage = ckpt["scheduler"]["current_stage"]
    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    print(f"[Resume] iter={ckpt['iteration']}  best_psnr={ckpt['best_psnr']:.4f} dB")
    return ckpt["iteration"], ckpt["best_psnr"]


# ══════════════════════════════════════════════════════════
# LOGGER (CSV + console)
# ══════════════════════════════════════════════════════════

class TrainLogger:
    def __init__(self, log_path: str):
        self.log_path = log_path
        self._fields  = ["iter", "loss", "psnr_train", "psnr_val", "lr", "time"]
        if not Path(log_path).exists():
            with open(log_path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=self._fields).writeheader()

    def log(self, row: dict):
        with open(self.log_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self._fields, extrasaction="ignore").writerow(row)
        print(
            f"[{row['iter']:>7}]  "
            f"loss={row.get('loss',0):.5f}  "
            f"psnr={row.get('psnr_train',0):.2f}dB  "
            f"val={row.get('psnr_val','-')!s:<7}  "
            f"lr={row.get('lr',0):.1e}  "
            f"t={row.get('time',0):.1f}s"
        )


# ══════════════════════════════════════════════════════════
# MAIN TRAINING LOOP
# ══════════════════════════════════════════════════════════

def train(args):
    # Воспроизводимость
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.benchmark = True

    os.makedirs(args.save_dir, exist_ok=True)

    # Датасеты
    train_dataset = DIV2KDataset(
        hr_dir=args.train_hr, scale=args.scale,
        patch_size=args.patch_size, augment=True, cache=args.cache,
    )
    val_dataset = DIV2KValDataset(hr_dir=args.val_hr, scale=args.scale)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers,
        pin_memory=True, drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1,
        shuffle=False, num_workers=2, pin_memory=True,
    )

    # Модель
    model = M23_RLFN(
        n_feats=args.n_feats, n_blocks=args.n_blocks,
        scale=args.scale, use_m23=True,
    ).to(args.device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] M23-RLFN | params={total_params/1e6:.3f}M | scale=×{args.scale}")

    # Loss + Optimizer + Scheduler
    criterion = CombinedSRLoss(freq_weight=args.freq_weight).to(args.device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=2e-4, betas=(0.9, 0.999), eps=1e-8,
    )
    total_iters = args.total_iters
    stages = [
        {"lr": 2e-4, "iters": total_iters // 3},
        {"lr": 1e-4, "iters": total_iters // 3},
        {"lr": 5e-5, "iters": total_iters - 2 * (total_iters // 3)},
    ]
    scheduler = WarmStartScheduler(optimizer, stages)
    scaler    = torch.amp.GradScaler(args.device) if args.device == "cuda" else None

    # Resume
    start_iter, best_psnr = 0, 0.0
    if args.resume and Path(args.resume).exists():
        start_iter, best_psnr = load_checkpoint(args.resume, model, optimizer, scheduler, scaler)

    logger = TrainLogger(os.path.join(args.save_dir, "train_log.csv"))

    print(f"\n{'='*60}")
    print(f"  Training M23-RLFN on DIV2K ×{args.scale}")
    print(f"  Device: {args.device.upper()}")
    print(f"  Total iters: {total_iters:,}  |  Batch: {args.batch_size}")
    print(f"{'='*60}\n")

    iteration   = start_iter
    loader_iter = iter(train_loader)
    t_start     = time.time()
    loss_accum  = 0.0
    psnr_accum  = 0.0

    while iteration < total_iters:
        try:
            lr_batch, hr_batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            lr_batch, hr_batch = next(loader_iter)

        lr_batch = lr_batch.to(args.device, non_blocking=True)
        hr_batch = hr_batch.to(args.device, non_blocking=True)

        # Forward + Backward
        model.train()
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.amp.autocast(args.device):
                pred = model(lr_batch)
                loss = criterion(pred, hr_batch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(lr_batch)
            loss = criterion(pred, hr_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()
        iteration += 1

        with torch.no_grad():
            psnr_batch = calc_psnr(pred.detach(), hr_batch, border=args.scale)
        loss_accum += loss.item()
        psnr_accum += psnr_batch

        # Логирование
        if iteration % args.log_every == 0:
            elapsed = time.time() - t_start
            logger.log({
                "iter":       iteration,
                "loss":       loss_accum / args.log_every,
                "psnr_train": psnr_accum / args.log_every,
                "psnr_val":   "-",
                "lr":         scheduler.current_lr,
                "time":       elapsed,
            })
            loss_accum = psnr_accum = 0.0
            t_start = time.time()

        # Валидация
        if iteration % args.val_every == 0:
            model.eval()
            val_psnr_sum = 0.0
            with torch.no_grad():
                for lr_v, hr_v in val_loader:
                    lr_v = lr_v.to(args.device)
                    hr_v = hr_v.to(args.device)
                    if scaler is not None:
                        with torch.amp.autocast(args.device):
                            pred_v = model(lr_v).clamp(0, 1)
                    else:
                        pred_v = model(lr_v).clamp(0, 1)
                    val_psnr_sum += calc_psnr(pred_v, hr_v, border=args.scale)
            val_psnr = val_psnr_sum / len(val_loader)
            print(f"\n>>> VAL PSNR: {val_psnr:.4f} dB  (best: {best_psnr:.4f} dB)")

            logger.log({
                "iter": iteration, "loss": "-", "psnr_train": "-",
                "psnr_val": round(val_psnr, 4), "lr": scheduler.current_lr, "time": 0,
            })

            ckpt_path = os.path.join(args.save_dir, f"m23_rlfn_iter{iteration}.pth")
            save_checkpoint(ckpt_path, model, optimizer, scheduler, iteration, val_psnr, scaler)

            if val_psnr > best_psnr:
                best_psnr = val_psnr
                best_path = os.path.join(args.save_dir, "m23_rlfn_best.pth")
                save_checkpoint(best_path, model, optimizer, scheduler, iteration, val_psnr, scaler)
                print(f"  ★ New best: {best_psnr:.4f} dB → {best_path}\n")

    print(f"\nTraining complete. Best PSNR: {best_psnr:.4f} dB")
    print(f"Best model: {os.path.join(args.save_dir, 'm23_rlfn_best.pth')}")


# ══════════════════════════════════════════════════════════
# INFERENCE HELPER
# ══════════════════════════════════════════════════════════

@torch.no_grad()
def upscale_image(
    model_path:  str,
    input_path:  str,
    output_path: str,
    scale:       int = 4,
    device:      str = "cuda",
    n_feats:     int = 52,
    n_blocks:    int = 8,
):
    """Апскейл одного изображения через обученную модель."""
    model = M23_RLFN(n_feats=n_feats, n_blocks=n_blocks, scale=scale)
    ckpt  = torch.load(model_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)

    lr = TF.to_tensor(Image.open(input_path).convert("RGB")).unsqueeze(0).to(device)
    if device == "cuda":
        with torch.amp.autocast(device):
            sr = model(lr).clamp(0, 1)
    else:
        sr = model(lr).clamp(0, 1)

    TF.to_pil_image(sr.squeeze(0).cpu()).save(output_path)
    print(f"Saved: {output_path}")


# ══════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="M23-RLFN DIV2K Training")
    # Данные
    p.add_argument("--train_hr",    default="data/DIV2K_train_HR")
    p.add_argument("--val_hr",      default="data/DIV2K_valid_HR")
    p.add_argument("--patch_size",  type=int,   default=256)
    p.add_argument("--cache",       action="store_true", default=True)
    # Модель
    p.add_argument("--scale",       type=int,   default=4)
    p.add_argument("--n_feats",     type=int,   default=52)
    p.add_argument("--n_blocks",    type=int,   default=8)
    # Тренировка
    p.add_argument("--total_iters", type=int,   default=300_000)
    p.add_argument("--batch_size",  type=int,   default=16)
    p.add_argument("--freq_weight", type=float, default=0.05)
    p.add_argument("--num_workers", type=int,   default=4)
    p.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed",        type=int,   default=42)
    # Логи и чекпоинты
    p.add_argument("--save_dir",    default="checkpoints/m23_rlfn")
    p.add_argument("--resume",      default=None)
    p.add_argument("--log_every",   type=int,   default=100)
    p.add_argument("--val_every",   type=int,   default=5_000)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
