"""DIV2K Dataset for training super-resolution models."""

import random
from pathlib import Path
from typing import Optional, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import functional as TF


class DIV2KDataset(Dataset):
    """
    DIV2K Training Dataset.

    High-quality image dataset for super-resolution training.
    Generates LR images on-the-fly via bicubic downsampling.

    Args:
        hr_dir: Path to high-resolution images
        scale: Downsampling scale factor
        patch_size: Size of random crops
        augment: Whether to apply augmentations
        cache: Whether to cache images in memory

    Example:
        >>> dataset = DIV2KDataset("data/DIV2K_train_HR", scale=4)
        >>> lr, hr = dataset[0]
    """

    def __init__(
        self,
        hr_dir: str,
        scale: int = 4,
        patch_size: int = 256,
        augment: bool = True,
        cache: bool = True,
    ):
        self.hr_dir = Path(hr_dir)
        self.scale = scale
        self.patch_size = patch_size
        self.augment = augment
        self.cache = cache

        # Find all images
        self.hr_paths: List[Path] = sorted(
            p for p in self.hr_dir.iterdir()
            if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp")
        )

        if not self.hr_paths:
            raise FileNotFoundError(f"No images found in {hr_dir}")

        self._cache: dict = {}

        print(f"[DIV2K] {len(self.hr_paths)} images | "
              f"scale=×{scale} | patch={patch_size}px | "
              f"augment={augment} | cache={cache}")

    def _load_hr(self, idx: int) -> torch.Tensor:
        """Load HR image, optionally from cache."""
        if self.cache and idx in self._cache:
            return self._cache[idx]

        img = Image.open(self.hr_paths[idx]).convert("RGB")
        tensor = TF.to_tensor(img)

        if self.cache:
            self._cache[idx] = tensor

        return tensor

    def _random_crop(self, hr: torch.Tensor) -> torch.Tensor:
        """Random crop from HR image."""
        _, H, W = hr.shape
        ps = self.patch_size

        if H < ps or W < ps:
            hr = TF.pad(hr, [0, 0, max(0, ps - W), max(0, ps - H)])
            _, H, W = hr.shape

        top = random.randint(0, H - ps)
        left = random.randint(0, W - ps)

        return hr[:, top:top + ps, left:left + ps]

    def _augment(self, hr: torch.Tensor) -> torch.Tensor:
        """Apply random augmentations."""
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
        # Load and process HR image
        hr = self._load_hr(idx)
        hr = self._random_crop(hr)

        if self.augment:
            hr = self._augment(hr)

        # Generate LR via bicubic downsampling
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
    """
    DIV2K Validation Dataset.

    Full images without augmentation for validation.

    Args:
        hr_dir: Path to validation images
        scale: Scale factor
        max_size: Maximum number of images
    """

    def __init__(
        self,
        hr_dir: str,
        scale: int = 4,
        max_size: int = 100,
    ):
        self.hr_paths = sorted(
            p for p in Path(hr_dir).iterdir()
            if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp")
        )[:max_size]
        self.scale = scale

    def __len__(self) -> int:
        return len(self.hr_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        hr = TF.to_tensor(Image.open(self.hr_paths[idx]).convert("RGB"))

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
