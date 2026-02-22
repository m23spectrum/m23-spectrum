"""
Inference utilities for M23-Spectrum
=====================================

High-level inference API with:
- Simple one-line inference
- Test-Time Augmentation (TTA) for +0.1-0.3 dB boost
- Batch processing
- Memory-efficient inference
"""

from typing import Union, Optional, List, Tuple
from pathlib import Path
import math

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

from .models import M23RLFN, create_model
from .utils.image import load_image, save_image, tensor_to_image, image_to_tensor
from .utils.device import get_device


class SuperResolver:
    """
    High-level super-resolution inference class.

    Provides easy-to-use API for image super-resolution with
    automatic model loading and inference.

    Args:
        model_name: Name of pre-trained model
        device: Device for inference ('cuda', 'cpu', or 'auto')
        half: Whether to use FP16 inference

    Example:
        >>> resolver = SuperResolver("m23-rlfn-x4")
        >>> sr_image = resolver.upscale("input.png", "output.png")
        >>> # Or process PIL Image directly
        >>> sr_pil = resolver.process(pil_image)
    """

    def __init__(
        self,
        model_name: str = "m23-rlfn-x4",
        device: str = "auto",
        half: bool = True,
    ):
        if device == "auto":
            device = get_device()

        self.device = device
        self.half = half and device.startswith("cuda")

        # Load model
        self.model = create_model(model_name, pretrained=True, device=device)
        self.model.eval()

        if self.half:
            self.model = self.model.half()

        self.model_name = model_name
        self.scale = getattr(self.model, 'scale', 4)

    @torch.no_grad()
    def process(
        self,
        image: Union[str, Path, Image.Image, torch.Tensor],
    ) -> Image.Image:
        """
        Process a single image.

        Args:
            image: Input image (path, PIL Image, or tensor)

        Returns:
            Super-resolved PIL Image
        """
        # Load image
        if isinstance(image, (str, Path)):
            tensor = load_image(image, to_tensor=True, device=self.device)
        elif isinstance(image, Image.Image):
            tensor = image_to_tensor(image, device=self.device)
        elif isinstance(image, torch.Tensor):
            tensor = image.to(self.device)
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        # Pad to multiple of 8 for best results
        orig_size = tensor.shape[2:]
        pad_h = (8 - orig_size[0] % 8) % 8
        pad_w = (8 - orig_size[1] % 8) % 8

        if pad_h > 0 or pad_w > 0:
            tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")

        # Inference
        if self.half:
            tensor = tensor.half()

        output = self.model(tensor)

        # Remove padding
        new_h = orig_size[0] * self.scale
        new_w = orig_size[1] * self.scale
        output = output[:, :, :new_h, :new_w]

        return tensor_to_image(output.float())

    def upscale(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
    ) -> Image.Image:
        """
        Upscale an image from file and save result.

        Args:
            input_path: Path to input image
            output_path: Path to save output

        Returns:
            Super-resolved PIL Image
        """
        sr_image = self.process(input_path)
        sr_image.save(output_path, quality=95)
        return sr_image

    def batch_process(
        self,
        images: List[Union[str, Path, Image.Image]],
        batch_size: int = 4,
    ) -> List[Image.Image]:
        """
        Process multiple images efficiently.

        Args:
            images: List of input images
            batch_size: Batch size for inference

        Returns:
            List of super-resolved PIL Images
        """
        results = []

        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]

            # Load batch
            tensors = []
            for img in batch:
                if isinstance(img, (str, Path)):
                    tensors.append(load_image(img, to_tensor=True, device=self.device))
                elif isinstance(img, Image.Image):
                    tensors.append(image_to_tensor(img, device=self.device))
                else:
                    tensors.append(img.to(self.device))

            # Stack into batch
            batch_tensor = torch.cat(tensors, dim=0)

            # Inference
            with torch.no_grad():
                if self.half:
                    batch_tensor = batch_tensor.half()
                outputs = self.model(batch_tensor)

            # Convert back to PIL
            for j in range(outputs.shape[0]):
                results.append(tensor_to_image(outputs[j:j+1].float()))

        return results


class TTAUpscaler:
    """
    Test-Time Augmentation Upscaler.

    Uses multiple augmented versions of the input for free PSNR boost.
    Typically adds +0.1-0.3 dB without any training changes.

    Augmentations used:
    - Horizontal flip
    - Vertical flip
    - 90°, 180°, 270° rotation
    - Transpose

    Args:
        model_name: Name of pre-trained model
        device: Device for inference
        tta_mode: TTA mode ('basic', 'full', 'flip_only')

    Example:
        >>> upscaler = TTAUpscaler("m23-rlfn-x4", tta_mode='full')
        >>> sr_image = upscaler.upscale("input.png")
    """

    def __init__(
        self,
        model_name: str = "m23-rlfn-x4",
        device: str = "auto",
        tta_mode: str = "full",
    ):
        if device == "auto":
            device = get_device()

        self.device = device
        self.model = create_model(model_name, pretrained=True, device=device)
        self.model.eval()
        self.scale = getattr(self.model, 'scale', 4)
        self.tta_mode = tta_mode

        # Define augmentations
        self.augmentations = self._get_augmentations(tta_mode)

    def _get_augmentations(self, mode: str) -> List[Tuple[str, callable]]:
        """Get list of augmentations for TTA."""
        augmentations = []

        if mode == "flip_only":
            augmentations = [
                ("identity", lambda x: x),
                ("hflip", lambda x: torch.flip(x, dims=[3])),
            ]
        elif mode == "basic":
            augmentations = [
                ("identity", lambda x: x),
                ("hflip", lambda x: torch.flip(x, dims=[3])),
                ("vflip", lambda x: torch.flip(x, dims=[2])),
                ("hvflip", lambda x: torch.flip(x, dims=[2, 3])),
            ]
        elif mode == "full":
            augmentations = [
                ("identity", lambda x: x),
                ("hflip", lambda x: torch.flip(x, dims=[3])),
                ("vflip", lambda x: torch.flip(x, dims=[2])),
                ("hvflip", lambda x: torch.flip(x, dims=[2, 3])),
                ("rot90", lambda x: torch.rot90(x, k=1, dims=[2, 3])),
                ("rot180", lambda x: torch.rot90(x, k=2, dims=[2, 3])),
                ("rot270", lambda x: torch.rot90(x, k=3, dims=[2, 3])),
            ]
        else:
            raise ValueError(f"Unknown TTA mode: {mode}")

        return augmentations

    def _reverse_augmentation(self, tensor: torch.Tensor, name: str) -> torch.Tensor:
        """Reverse the augmentation applied to output."""
        if name == "identity":
            return tensor
        elif name == "hflip":
            return torch.flip(tensor, dims=[3])
        elif name == "vflip":
            return torch.flip(tensor, dims=[2])
        elif name == "hvflip":
            return torch.flip(tensor, dims=[2, 3])
        elif name == "rot90":
            return torch.rot90(tensor, k=-1, dims=[2, 3])
        elif name == "rot180":
            return torch.rot90(tensor, k=-2, dims=[2, 3])
        elif name == "rot270":
            return torch.rot90(tensor, k=-3, dims=[2, 3])
        return tensor

    @torch.no_grad()
    def process(
        self,
        image: Union[str, Path, Image.Image, torch.Tensor],
    ) -> Image.Image:
        """
        Process image with TTA.

        Args:
            image: Input image

        Returns:
            Super-resolved PIL Image (average of all TTA predictions)
        """
        # Load image
        if isinstance(image, (str, Path)):
            tensor = load_image(image, to_tensor=True, device=self.device)
        elif isinstance(image, Image.Image):
            tensor = image_to_tensor(image, device=self.device)
        elif isinstance(image, torch.Tensor):
            tensor = image.to(self.device)
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        # Apply TTA
        outputs = []

        for name, aug_fn in self.augmentations:
            # Apply augmentation
            augmented = aug_fn(tensor)

            # Inference
            output = self.model(augmented)

            # Reverse augmentation
            output = self._reverse_augmentation(output, name)

            outputs.append(output)

        # Average predictions
        avg_output = torch.stack(outputs).mean(dim=0)

        return tensor_to_image(avg_output)

    def upscale(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
    ) -> Image.Image:
        """
        Upscale with TTA and optionally save.

        Args:
            input_path: Path to input image
            output_path: Optional path to save output

        Returns:
            Super-resolved PIL Image
        """
        sr_image = self.process(input_path)

        if output_path is not None:
            sr_image.save(output_path, quality=95)

        return sr_image


def upscale_image(
    input_path: str,
    output_path: str,
    model_name: str = "m23-rlfn-x4",
    device: str = "auto",
    tta: bool = False,
) -> None:
    """
    Convenience function for quick image upscaling.

    Args:
        input_path: Path to input image
        output_path: Path to save output
        model_name: Name of model to use
        device: Device for inference
        tta: Whether to use TTA

    Example:
        >>> upscale_image("low_res.png", "high_res.png", model_name="m23-rlfn-x4")
    """
    if tta:
        upscaler = TTAUpscaler(model_name, device=device)
    else:
        upscaler = SuperResolver(model_name, device=device)

    upscaler.upscale(input_path, output_path)
    print(f"Saved: {output_path}")
