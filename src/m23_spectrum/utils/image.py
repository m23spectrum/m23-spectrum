"""Image utility functions."""

from typing import Union, Optional, Tuple
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def load_image(
    path: Union[str, Path],
    mode: str = "RGB",
    to_tensor: bool = True,
    device: str = "cpu",
) -> Union[np.ndarray, torch.Tensor]:
    """
    Load an image from file.

    Args:
        path: Path to image file
        mode: Color mode ('RGB', 'L', etc.)
        to_tensor: Whether to convert to PyTorch tensor
        device: Device for tensor

    Returns:
        Image as numpy array (H, W, C) or tensor (1, C, H, W)
    """
    img = Image.open(path).convert(mode)

    if to_tensor:
        return image_to_tensor(img, device=device)
    else:
        return np.array(img)


def save_image(
    tensor: Union[torch.Tensor, np.ndarray],
    path: Union[str, Path],
    quality: int = 95,
) -> None:
    """
    Save a tensor or array as an image.

    Args:
        tensor: Image tensor (1, C, H, W) or array (H, W, C)
        path: Output path
        quality: JPEG quality (for JPEG files)
    """
    img = tensor_to_image(tensor)
    img.save(path, quality=quality)


def tensor_to_image(
    tensor: Union[torch.Tensor, np.ndarray],
) -> Image.Image:
    """
    Convert tensor to PIL Image.

    Args:
        tensor: Image tensor (B, C, H, W) or (C, H, W) or array

    Returns:
        PIL Image
    """
    if isinstance(tensor, torch.Tensor):
        # Handle batch dimension
        if tensor.dim() == 4:
            tensor = tensor[0]

        # Move to CPU and convert to numpy
        tensor = tensor.detach().cpu().numpy()

    # (C, H, W) -> (H, W, C)
    if tensor.ndim == 3:
        tensor = tensor.transpose(1, 2, 0)

    # Clip to [0, 1] and convert to uint8
    tensor = np.clip(tensor, 0, 1)
    tensor = (tensor * 255).astype(np.uint8)

    return Image.fromarray(tensor)


def image_to_tensor(
    image: Union[Image.Image, np.ndarray],
    device: str = "cpu",
) -> torch.Tensor:
    """
    Convert PIL Image or numpy array to tensor.

    Args:
        image: PIL Image or numpy array
        device: Target device

    Returns:
        Tensor of shape (1, C, H, W) in range [0, 1]
    """
    if isinstance(image, Image.Image):
        array = np.array(image)
    else:
        array = image

    # (H, W, C) -> (C, H, W)
    if array.ndim == 3:
        array = array.transpose(2, 0, 1)

    # Normalize to [0, 1]
    tensor = torch.from_numpy(array.astype(np.float32) / 255.0)

    # Add batch dimension
    tensor = tensor.unsqueeze(0)

    return tensor.to(device)


def pad_to_multiple(
    tensor: torch.Tensor,
    multiple: int = 8,
    mode: str = "reflect",
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Pad tensor to be divisible by a multiple.

    Args:
        tensor: Input tensor (B, C, H, W)
        multiple: Target multiple
        mode: Padding mode

    Returns:
        Padded tensor and original size
    """
    _, _, h, w = tensor.shape

    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple

    if pad_h > 0 or pad_w > 0:
        if mode == "reflect":
            tensor = torch.nn.functional.pad(
                tensor, (0, pad_w, 0, pad_h), mode="reflect"
            )
        else:
            tensor = torch.nn.functional.pad(
                tensor, (0, pad_w, 0, pad_h), mode="constant", value=0
            )

    return tensor, (h, w)


def unpad(
    tensor: torch.Tensor,
    original_size: Tuple[int, int],
) -> torch.Tensor:
    """
    Remove padding from tensor.

    Args:
        tensor: Padded tensor
        original_size: Original (height, width)

    Returns:
        Unpadded tensor
    """
    h, w = original_size
    return tensor[:, :, :h, :w]
