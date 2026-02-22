"""Standard benchmark datasets for super-resolution evaluation."""

from pathlib import Path
from typing import List, Tuple, Optional
import hashlib

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import functional as TF


# Standard benchmark dataset URLs
DATASET_URLS = {
    "set5": "https://github.com/m23spectrum/m23-spectrum/releases/download/datasets/Set5.zip",
    "set14": "https://github.com/m23spectrum/m23-spectrum/releases/download/datasets/Set14.zip",
    "bsd100": "https://github.com/m23spectrum/m23-spectrum/releases/download/datasets/BSD100.zip",
    "urban100": "https://github.com/m23spectrum/m23-spectrum/releases/download/datasets/Urban100.zip",
    "manga109": "https://github.com/m23spectrum/m23-spectrum/releases/download/datasets/Manga109.zip",
}


class BenchmarkDataset(Dataset):
    """
    Base class for benchmark datasets.

    Args:
        hr_dir: Path to HR images
        scale: Scale factor
        name: Dataset name
    """

    def __init__(
        self,
        hr_dir: str,
        scale: int = 4,
        name: str = "benchmark",
    ):
        self.hr_dir = Path(hr_dir)
        self.scale = scale
        self.name = name

        self.hr_paths = sorted(
            p for p in self.hr_dir.iterdir()
            if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp")
        )

        if not self.hr_paths:
            raise FileNotFoundError(f"No images found in {hr_dir}")

        print(f"[{name}] {len(self.hr_paths)} images | scale=×{scale}")

    def __len__(self) -> int:
        return len(self.hr_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        hr = TF.to_tensor(Image.open(self.hr_paths[idx]).convert("RGB"))

        # Ensure dimensions are divisible by scale
        _, H, W = hr.shape
        H = (H // self.scale) * self.scale
        W = (W // self.scale) * self.scale
        hr = hr[:, :H, :W]

        # Generate LR
        lr = F.interpolate(
            hr.unsqueeze(0),
            size=(H // self.scale, W // self.scale),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        ).squeeze(0).clamp(0, 1)

        return lr, hr, self.hr_paths[idx].stem


class Set5Dataset(BenchmarkDataset):
    """
    Set5 benchmark dataset.

    Small dataset (5 images) for quick evaluation.
    Standard benchmark for SR papers.

    Args:
        hr_dir: Path to Set5 HR images
        scale: Scale factor
    """

    def __init__(self, hr_dir: str, scale: int = 4):
        super().__init__(hr_dir, scale, "Set5")


class Set14Dataset(BenchmarkDataset):
    """
    Set14 benchmark dataset.

    Medium dataset (14 images) for evaluation.

    Args:
        hr_dir: Path to Set14 HR images
        scale: Scale factor
    """

    def __init__(self, hr_dir: str, scale: int = 4):
        super().__init__(hr_dir, scale, "Set14")


class BSD100Dataset(BenchmarkDataset):
    """
    BSD100 benchmark dataset.

    Large dataset (100 images) from BSD500.

    Args:
        hr_dir: Path to BSD100 HR images
        scale: Scale factor
    """

    def __init__(self, hr_dir: str, scale: int = 4):
        super().__init__(hr_dir, scale, "BSD100")


class Urban100Dataset(BenchmarkDataset):
    """
    Urban100 benchmark dataset.

    100 urban images with repetitive structures.
    Good for testing texture preservation.

    Args:
        hr_dir: Path to Urban100 HR images
        scale: Scale factor
    """

    def __init__(self, hr_dir: str, scale: int = 4):
        super().__init__(hr_dir, scale, "Urban100")


class Manga109Dataset(BenchmarkDataset):
    """
    Manga109 benchmark dataset.

    109 manga images for anime/manga SR evaluation.

    Args:
        hr_dir: Path to Manga109 HR images
        scale: Scale factor
    """

    def __init__(self, hr_dir: str, scale: int = 4):
        super().__init__(hr_dir, scale, "Manga109")


def evaluate_model(
    model: torch.nn.Module,
    dataset: BenchmarkDataset,
    device: str = "cuda",
    save_dir: Optional[str] = None,
) -> dict:
    """
    Evaluate a model on a benchmark dataset.

    Args:
        model: SR model to evaluate
        dataset: Benchmark dataset
        device: Device for inference
        save_dir: Optional directory to save outputs

    Returns:
        Dictionary with PSNR and SSIM results
    """
    from ..utils.metrics import calculate_psnr, calculate_ssim

    model.eval()
    model.to(device)

    psnr_values = []
    ssim_values = []

    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for idx in range(len(dataset)):
            lr, hr, name = dataset[idx]
            lr = lr.unsqueeze(0).to(device)

            # Inference
            sr = model(lr).clamp(0, 1)

            # Calculate metrics (with border)
            border = dataset.scale
            psnr = calculate_psnr(sr, hr.unsqueeze(0).to(device), border=border)
            ssim = calculate_ssim(sr, hr.unsqueeze(0).to(device), border=border)

            psnr_values.append(psnr)
            ssim_values.append(ssim)

            # Save output
            if save_dir:
                sr_image = TF.to_pil_image(sr.squeeze(0).cpu())
                sr_image.save(save_path / f"{name}_x{dataset.scale}.png")

    return {
        "dataset": dataset.name,
        "scale": dataset.scale,
        "psnr_mean": sum(psnr_values) / len(psnr_values),
        "ssim_mean": sum(ssim_values) / len(ssim_values),
        "psnr_values": psnr_values,
        "ssim_values": ssim_values,
        "num_images": len(dataset),
    }
