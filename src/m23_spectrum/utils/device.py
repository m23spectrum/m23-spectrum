"""Device utilities."""

from typing import Optional
import torch


def get_device() -> str:
    """
    Get the best available device.

    Returns:
        Device string ('cuda', 'mps', or 'cpu')
    """
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def to_device(
    obj,
    device: Optional[str] = None,
):
    """
    Move object to device.

    Args:
        obj: Tensor, module, or dict/list of tensors
        device: Target device (default: auto-detect)

    Returns:
        Object on target device
    """
    if device is None:
        device = get_device()

    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, torch.nn.Module):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(to_device(v, device) for v in obj)
    else:
        return obj


def get_device_info() -> dict:
    """
    Get device information.

    Returns:
        Dictionary with device information
    """
    info = {
        "device": get_device(),
        "cuda_available": torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        info.update({
            "cuda_device_name": torch.cuda.get_device_name(0),
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_memory_allocated": torch.cuda.memory_allocated(0) / 1024**3,
            "cuda_memory_reserved": torch.cuda.memory_reserved(0) / 1024**3,
        })

    return info


def print_device_info():
    """Print formatted device information."""
    info = get_device_info()

    print("=" * 50)
    print("  Device Information")
    print("=" * 50)
    print(f"  Device: {info['device'].upper()}")
    print(f"  CUDA Available: {info['cuda_available']}")

    if info['cuda_available']:
        print(f"  GPU: {info['cuda_device_name']}")
        print(f"  GPU Count: {info['cuda_device_count']}")
        print(f"  Memory Allocated: {info['cuda_memory_allocated']:.2f} GB")
        print(f"  Memory Reserved: {info['cuda_memory_reserved']:.2f} GB")

    print("=" * 50)
