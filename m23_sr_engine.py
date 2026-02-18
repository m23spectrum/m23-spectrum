"""
M23-Spectrum SR Engine v3.0
===========================

Полный стек для прыжка к 29-30 dB:

1. M23SpectrumInit     — инициализация весов (v2, исправленная)
2. RLFB                — Residual Local Feature Block (NTIRE 2022 winner)
3. M23_RLFN            — полная сеть: RLFN + M23-инициализация + PixelShuffle
4. CharbonnierLoss     — робастный L1 (лучше MSE для SR)
5. FrequencyLoss       — FFT-loss на амплитудном спектре (CVPR 2024)
6. CombinedSRLoss      — L1 + Freq (оптимальная комбинация)
7. WarmStartScheduler  — multi-stage warm-start (стратегия RLFN paper)
8. SRTrainer           — полный тренировочный цикл с AMP

Ориентир: ~29-30 dB на Set5 x4 при ~900K параметрах на RTX 4070 Ti Super.
"""

import numpy as np
import warnings
import threading
from typing import Tuple, Optional, Dict, List

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.fft
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not found. Only numpy API available.", ImportWarning)


# ══════════════════════════════════════════════════════════
# 1. M23 SPECTRUM INITIALIZER (v2 core)
# ══════════════════════════════════════════════════════════

def _build_m23_base_spectrum() -> np.ndarray:
    elkies = np.roots([1, 1, 9, -10, 8])
    parts = [elkies]
    for g in elkies:
        parts.append(np.roots([1, -g, 1]))            # P2
        parts.append(np.roots([1, g, 0, -1]))          # P3
        parts.append(np.roots([1, g, 0, -(g**2+1), -g]))  # P4
    return np.concatenate(parts)  # 40 компонент


_M23_BASE   = _build_m23_base_spectrum()
_M23_BASE_N = len(_M23_BASE)   # 40
_CACHE: Dict[Tuple, np.ndarray] = {}
_CACHE_LOCK = threading.RLock()


def m23_spectrum(fan_in: int, seed: Optional[int] = None) -> np.ndarray:
    """
    Строит спектр M23 размером fan_in.
    Результат нормирован: max|λ| ≈ 1/sqrt(fan_in).
    """
    key = (fan_in, seed)
    with _CACHE_LOCK:
        if key in _CACHE:
            return _CACHE[key].copy()

    if fan_in <= _M23_BASE_N:
        idx  = np.round(np.linspace(0, _M23_BASE_N - 1, fan_in)).astype(int)
        spec = _M23_BASE[idx].copy()
    else:
        n_tiles = (fan_in + _M23_BASE_N - 1) // _M23_BASE_N
        tiles   = [_M23_BASE * np.exp(1j * 2 * np.pi * k / n_tiles) for k in range(n_tiles)]
        spec    = np.concatenate(tiles)[:fan_in]

    if seed is not None:
        rng   = np.random.default_rng(seed)
        spec += 1e-4 * (rng.standard_normal(fan_in) + 1j * rng.standard_normal(fan_in))

    norm = np.max(np.abs(spec)) or 1e-10
    spec = spec / norm / np.sqrt(max(fan_in, 1))

    with _CACHE_LOCK:
        _CACHE[key] = spec.copy()
    return spec


def m23_init_tensor(
    tensor: "torch.Tensor",
    variant: str = "orthogonal",
    seed: Optional[int] = None,
) -> "torch.Tensor":
    """
    Инициализирует PyTorch-тензор весов через M23-Spectrum.

    variant:
        'orthogonal'  — SVD-ортогональная (рекомендуется для SR)
        'standard'    — QR-ортогональная
        'scaled'      — Xavier-like масштаб
        'transformer' — 1/sqrt(d_model)
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch required")

    shape = tuple(tensor.shape)
    ndim  = len(shape)

    if ndim == 1:
        fan_in = fan_out = shape[0]
    elif ndim == 2:
        fan_out, fan_in = shape
    else:  # Conv: (C_out, C_in, *kernel)
        fan_out = shape[0]
        fan_in  = int(np.prod(shape[1:]))

    spec = m23_spectrum(fan_in, seed=seed)

    # Строим матрицу (fan_out × fan_in) с фазовой структурой
    mat = np.zeros((fan_out, fan_in), dtype=np.complex128)
    for col in range(fan_in):
        phase = np.exp(1j * 2 * np.pi * col / fan_in)
        for row in range(fan_out):
            mat[row, col] = spec[(row + col) % len(spec)] * (phase ** col)

    if variant == "orthogonal":
        try:
            U, _, Vt = np.linalg.svd(mat, full_matrices=False)
            mat = U @ Vt
        except np.linalg.LinAlgError:
            Q, R = np.linalg.qr(mat)
            mat  = Q * np.sign(np.diagonal(R))
    elif variant == "standard":
        Q, R = np.linalg.qr(mat)
        mat  = Q * np.sign(np.diagonal(R)[:mat.shape[1]])
    elif variant == "scaled":
        mat = mat * np.sqrt(2.0 / (fan_in + fan_out))
    elif variant == "transformer":
        mat = mat / np.sqrt(fan_in)

    weights = np.real(mat).reshape(shape).astype(np.float32)
    with torch.no_grad():
        tensor.copy_(torch.from_numpy(weights))
    return tensor


# ══════════════════════════════════════════════════════════
# 2. RESIDUAL LOCAL FEATURE BLOCK (RLFB — NTIRE 2022)
# ══════════════════════════════════════════════════════════

if TORCH_AVAILABLE:

    class ESA(nn.Module):
        """Enhanced Spatial Attention из RLFN (Liu et al., NTIRE 2022)."""

        def __init__(self, n_feats: int):
            super().__init__()
            f = n_feats // 4
            self.conv1    = nn.Conv2d(n_feats, f, 1)
            self.conv_f   = nn.Conv2d(f, f, 1)
            self.conv_max = nn.Conv2d(f, f, 3, padding=1)
            self.conv2    = nn.Conv2d(f, f, 3, stride=2, padding=0)
            self.conv3    = nn.Conv2d(f, f, 3, padding=1)
            self.conv3_   = nn.Conv2d(f, f, 3, padding=1)
            self.conv4    = nn.Conv2d(f, n_feats, 1)
            self.sigmoid  = nn.Sigmoid()
            self.relu     = nn.ReLU(inplace=True)

        def forward(self, x):
            c1_    = self.conv1(x)
            c1     = self.conv2(c1_)
            v_max  = F.max_pool2d(c1, kernel_size=7, stride=3)
            v_range= self.relu(self.conv_max(v_max))
            c3     = self.relu(self.conv3(v_range))
            c3     = self.conv3_(c3)
            c3     = F.interpolate(c3, (x.size(2), x.size(3)),
                                   mode='bilinear', align_corners=False)
            cf     = self.conv_f(c1_)
            c4     = self.conv4(c3 + cf)
            m      = self.sigmoid(c4)
            return x * m


    class RLFB(nn.Module):
        """Residual Local Feature Block — три свёртки + ESA + residual."""

        def __init__(self, n_feats: int, use_m23: bool = True):
            super().__init__()
            self.conv1 = nn.Conv2d(n_feats, n_feats, 3, padding=1)
            self.conv2 = nn.Conv2d(n_feats, n_feats, 3, padding=1)
            self.conv3 = nn.Conv2d(n_feats, n_feats, 3, padding=1)
            self.esa   = ESA(n_feats)
            self.act   = nn.GELU()
            if use_m23:
                self._apply_m23_init()

        def _apply_m23_init(self):
            for _, m in self.named_modules():
                if isinstance(m, nn.Conv2d):
                    m23_init_tensor(m.weight, variant="orthogonal")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        def forward(self, x):
            out = self.act(self.conv1(x))
            out = self.act(self.conv2(out))
            out = self.conv3(out)
            out = self.esa(out)
            return out + x  # residual


    # ══════════════════════════════════════════════════════════
    # 3. M23-RLFN — полная SR-сеть
    # ══════════════════════════════════════════════════════════

    class M23_RLFN(nn.Module):
        """
        M23-RLFN: RLFN архитектура + M23-Spectrum инициализация.

        Параметры по умолчанию (~900K параметров):
            n_feats=52, n_blocks=8, scale=4

        Ориентир: 29-30 dB на Set5 x4 при обучении на DIV2K.
        """

        def __init__(
            self,
            in_channels:  int = 3,
            out_channels: int = 3,
            n_feats:      int = 52,
            n_blocks:     int = 8,
            scale:        int = 4,
            use_m23:      bool = True,
        ):
            super().__init__()
            self.scale = scale

            # Head: первичное извлечение признаков
            self.head = nn.Conv2d(in_channels, n_feats, 3, padding=1)

            # Body: стек RLFB
            self.body = nn.Sequential(
                *[RLFB(n_feats, use_m23=use_m23) for _ in range(n_blocks)]
            )

            # После body — свёртка для агрегации
            self.body_conv = nn.Conv2d(n_feats, n_feats, 3, padding=1)

            # Upsampling: PixelShuffle (sub-pixel convolution)
            self.upsample = nn.Sequential(
                nn.Conv2d(n_feats, out_channels * (scale ** 2), 3, padding=1),
                nn.PixelShuffle(scale),
            )

            if use_m23:
                self._init_remaining()

        def _init_remaining(self):
            for name, m in self.named_modules():
                if isinstance(m, nn.Conv2d):
                    # RLFB уже инициализированы сами
                    if not any(f"body.{i}." in name for i in range(20)):
                        m23_init_tensor(m.weight, variant="orthogonal")
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)

        def forward(self, x):
            # Bilinear upscale LR → HR размер (global residual)
            bicubic  = F.interpolate(x, scale_factor=self.scale,
                                     mode='bicubic', align_corners=False)
            feat     = self.head(x)
            body_out = self.body(feat)
            body_out = self.body_conv(body_out) + feat  # global body residual
            out      = self.upsample(body_out)
            return out + bicubic  # SR = detail + bicubic


    # ══════════════════════════════════════════════════════════
    # 4. LOSS FUNCTIONS
    # ══════════════════════════════════════════════════════════

    class CharbonnierLoss(nn.Module):
        """L1 Charbonnier: sqrt((x-y)^2 + eps^2). Робастнее MSE для SR."""

        def __init__(self, eps: float = 1e-3):
            super().__init__()
            self.eps2 = eps ** 2

        def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps2))


    class FrequencyLoss(nn.Module):
        """
        FFT Amplitude + Phase Loss.
        Штрафует за несоответствие высокочастотных компонент.
        Реализует идею из CVPR 2024 FDL.
        """

        def __init__(self, loss_weight: float = 0.1):
            super().__init__()
            self.weight = loss_weight

        def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            pred_fft   = torch.fft.rfft2(pred,   norm="ortho")
            target_fft = torch.fft.rfft2(target, norm="ortho")

            amp_loss   = F.l1_loss(torch.abs(pred_fft), torch.abs(target_fft))
            phase_loss = (1 - torch.cos(
                torch.angle(pred_fft) - torch.angle(target_fft)
            )).mean()

            return self.weight * (amp_loss + 0.1 * phase_loss)


    class CombinedSRLoss(nn.Module):
        """
        L = Charbonnier + λ_freq * FrequencyLoss
        Рекомендуемые значения: freq_weight=0.05
        """

        def __init__(self, freq_weight: float = 0.05):
            super().__init__()
            self.charb = CharbonnierLoss(eps=1e-3)
            self.freq  = FrequencyLoss(loss_weight=freq_weight)

        def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            return self.charb(pred, target) + self.freq(pred, target)


    # ══════════════════════════════════════════════════════════
    # 5. WARM-START MULTI-STAGE SCHEDULER
    # ══════════════════════════════════════════════════════════

    class WarmStartScheduler:
        """
        Multi-stage warm-start стратегия из RLFN paper.

        stages = [
            {"lr": 2e-4, "iters": 100_000},
            {"lr": 1e-4, "iters": 100_000},
            {"lr": 5e-5, "iters": 100_000},
        ]
        """

        def __init__(self, optimizer, stages: List[Dict]):
            self.optimizer     = optimizer
            self.stages        = stages
            self.current_iter  = 0
            self.current_stage = 0
            self._set_lr(stages[0]["lr"])

        def _set_lr(self, lr: float):
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

        def step(self):
            self.current_iter += 1
            stage = self.stages[self.current_stage]
            if self.current_iter >= stage["iters"]:
                self.current_stage = min(self.current_stage + 1, len(self.stages) - 1)
                if self.current_stage < len(self.stages):
                    self._set_lr(self.stages[self.current_stage]["lr"])
                self.current_iter = 0

        @property
        def current_lr(self) -> float:
            return self.optimizer.param_groups[0]["lr"]


    # ══════════════════════════════════════════════════════════
    # 6. SRTRAINER
    # ══════════════════════════════════════════════════════════

    class SRTrainer:
        """
        Полный тренировочный цикл:
        - AMP (Automatic Mixed Precision) для RTX 4070 Ti Super
        - Gradient clipping (max_norm=1.0)
        - Warm-start scheduler
        - PSNR мониторинг
        """

        def __init__(
            self,
            model:       "M23_RLFN",
            device:      str = "cuda",
            freq_weight: float = 0.05,
            stages:      Optional[List[Dict]] = None,
        ):
            self.model     = model.to(device)
            self.device    = device
            self.criterion = CombinedSRLoss(freq_weight=freq_weight)
            self.optimizer = torch.optim.Adam(
                model.parameters(), lr=2e-4,
                betas=(0.9, 0.999), eps=1e-8,
            )
            if stages is None:
                stages = [
                    {"lr": 2e-4, "iters": 100_000},
                    {"lr": 1e-4, "iters": 100_000},
                    {"lr": 5e-5, "iters": 100_000},
                ]
            self.scheduler = WarmStartScheduler(self.optimizer, stages)
            self.scaler    = torch.amp.GradScaler(device) if device == "cuda" else None

        @staticmethod
        def calc_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
            with torch.no_grad():
                mse = F.mse_loss(pred.clamp(0, 1), target.clamp(0, 1))
                if mse == 0:
                    return float("inf")
                return -10 * torch.log10(mse).item()

        def train_step(self, lr_batch: torch.Tensor, hr_batch: torch.Tensor) -> Dict:
            self.model.train()
            lr_batch = lr_batch.to(self.device)
            hr_batch = hr_batch.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)

            if self.scaler is not None:
                with torch.amp.autocast(self.device):
                    pred = self.model(lr_batch)
                    loss = self.criterion(pred, hr_batch)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                pred = self.model(lr_batch)
                loss = self.criterion(pred, hr_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            self.scheduler.step()
            return {
                "loss": loss.item(),
                "psnr": self.calc_psnr(pred.detach(), hr_batch),
                "lr":   self.scheduler.current_lr,
            }

        @torch.no_grad()
        def evaluate(self, val_loader) -> float:
            self.model.eval()
            psnr_sum, count = 0.0, 0
            for lr_batch, hr_batch in val_loader:
                lr_batch = lr_batch.to(self.device)
                hr_batch = hr_batch.to(self.device)
                pred     = self.model(lr_batch).clamp(0, 1)
                psnr_sum += self.calc_psnr(pred, hr_batch)
                count    += 1
            return psnr_sum / max(count, 1)


# ══════════════════════════════════════════════════════════
# 7. QUICK USAGE EXAMPLE
# ══════════════════════════════════════════════════════════
"""
Пример использования:

    from m23_sr_engine import M23_RLFN, SRTrainer

    model   = M23_RLFN(n_feats=52, n_blocks=8, scale=4)
    trainer = SRTrainer(model, device="cuda", freq_weight=0.05)

    for lr_img, hr_img in train_loader:
        stats = trainer.train_step(lr_img, hr_img)
        print(f"loss={stats['loss']:.4f}  psnr={stats['psnr']:.2f}dB")

    # Инференс:
    model.eval()
    with torch.no_grad():
        sr = model(lr_tensor.cuda()).clamp(0, 1)
"""
