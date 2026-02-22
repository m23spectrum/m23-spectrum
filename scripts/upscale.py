#!/usr/bin/env python3
"""
M23-Spectrum CLI - Super-Resolution Command Line Tool
======================================================

Usage:
    m23-upscale input.png output.png --model m23-rlfn-x4 --tta
    m23-benchmark --dataset set5 --model m23-rlfn-x4
    m23-train --config config.yaml
"""

import argparse
import sys
from pathlib import Path


def main_upscale():
    """CLI entry point for upscaling."""
    parser = argparse.ArgumentParser(
        description="M23-Spectrum Image Super-Resolution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "input",
        type=str,
        help="Input image path or directory",
    )

    parser.add_argument(
        "output",
        type=str,
        help="Output image path or directory",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="m23-rlfn-x4",
        choices=[
            "m23-rlfn-x2",
            "m23-rlfn-x3",
            "m23-rlfn-x4",
            "m23-rlfn-x4-large",
            "m23-swinir-x2",
            "m23-swinir-x4",
        ],
        help="Model to use for super-resolution",
    )

    parser.add_argument(
        "--tta",
        action="store_true",
        help="Use Test-Time Augmentation (+0.1-0.3 dB boost)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device for inference",
    )

    parser.add_argument(
        "--half",
        action="store_true",
        help="Use FP16 inference (faster on GPU)",
    )

    args = parser.parse_args()

    # Import here to avoid slow startup
    import torch
    from PIL import Image

    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from m23_spectrum import SuperResolver, TTAUpscaler

    input_path = Path(args.input)
    output_path = Path(args.output)

    # Check input exists
    if not input_path.exists():
        print(f"Error: Input not found: {input_path}")
        sys.exit(1)

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model: {args.model}")
    if args.tta:
        upscaler = TTAUpscaler(args.model, device=args.device)
    else:
        upscaler = SuperResolver(args.model, device=args.device, half=args.half)

    # Process
    if input_path.is_file():
        print(f"Processing: {input_path}")
        upscaler.upscale(input_path, output_path)
        print(f"Saved: {output_path}")
    else:
        # Batch processing
        output_path.mkdir(parents=True, exist_ok=True)

        image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
        images = [
            f for f in input_path.iterdir()
            if f.suffix.lower() in image_extensions
        ]

        print(f"Found {len(images)} images")

        for i, img_path in enumerate(images, 1):
            out_path = output_path / img_path.name
            print(f"[{i}/{len(images)}] Processing: {img_path.name}")
            upscaler.upscale(img_path, out_path)

        print(f"Done! Results saved to: {output_path}")


def main_benchmark():
    """CLI entry point for benchmarking."""
    parser = argparse.ArgumentParser(
        description="M23-Spectrum Benchmark Evaluation",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="set5",
        choices=["set5", "set14", "bsd100", "urban100", "all"],
        help="Dataset to evaluate on",
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/benchmark",
        help="Path to benchmark datasets",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="m23-rlfn-x4",
        help="Model to evaluate",
    )

    parser.add_argument(
        "--scale",
        type=int,
        default=4,
        choices=[2, 3, 4],
        help="Scale factor",
    )

    parser.add_argument(
        "--tta",
        action="store_true",
        help="Use TTA",
    )

    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Directory to save outputs",
    )

    args = parser.parse_args()

    # Import here
    import torch

    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from m23_spectrum import SuperResolver, TTAUpscaler
    from m23_spectrum.data.benchmark import (
        Set5Dataset, Set14Dataset, BSD100Dataset,
        Urban100Dataset, evaluate_model,
    )
    from m23_spectrum.utils.metrics import calculate_psnr

    # Load model
    print(f"Loading model: {args.model}")
    if args.tta:
        upscaler = TTAUpscaler(args.model, device="auto")
        model = upscaler.model
    else:
        upscaler = SuperResolver(args.model, device="auto")
        model = upscaler.model

    # Get datasets
    data_dir = Path(args.data_dir)

    datasets = []
    if args.dataset == "all":
        datasets = [
            ("Set5", Set5Dataset),
            ("Set14", Set14Dataset),
            ("BSD100", BSD100Dataset),
        ]
    elif args.dataset == "set5":
        datasets = [("Set5", Set5Dataset)]
    elif args.dataset == "set14":
        datasets = [("Set14", Set14Dataset)]
    elif args.dataset == "bsd100":
        datasets = [("BSD100", BSD100Dataset)]

    # Evaluate
    print("\n" + "=" * 60)
    print("  M23-Spectrum Benchmark Results")
    print("=" * 60)
    print(f"  Model: {args.model}")
    print(f"  Scale: ×{args.scale}")
    print(f"  TTA: {'Enabled' if args.tta else 'Disabled'}")
    print("=" * 60 + "\n")

    results = []

    for name, dataset_class in datasets:
        dataset_path = data_dir / name
        if not dataset_path.exists():
            print(f"Skipping {name}: not found at {dataset_path}")
            continue

        dataset = dataset_class(str(dataset_path), scale=args.scale)
        result = evaluate_model(model, dataset, save_dir=args.save_dir)

        results.append(result)
        print(f"  {name}: PSNR = {result['psnr_mean']:.2f} dB | "
              f"SSIM = {result['ssim_mean']:.4f}")

    if results:
        print("\n" + "-" * 60)
        avg_psnr = sum(r['psnr_mean'] for r in results) / len(results)
        avg_ssim = sum(r['ssim_mean'] for r in results) / len(results)
        print(f"  Average: PSNR = {avg_psnr:.2f} dB | SSIM = {avg_ssim:.4f}")
        print("-" * 60)


def main_info():
    """Print model and device information."""
    import torch

    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from m23_spectrum.models.factory import print_model_table
    from m23_spectrum.utils.device import print_device_info

    print("\n")
    print_model_table()
    print("\n")
    print_device_info()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        command = sys.argv[1]
        sys.argv = sys.argv[1:]

        if command == "upscale":
            main_upscale()
        elif command == "benchmark":
            main_benchmark()
        elif command == "info":
            main_info()
        else:
            print(f"Unknown command: {command}")
            print("Available commands: upscale, benchmark, info")
    else:
        print("M23-Spectrum CLI")
        print("Usage: python -m m23_spectrum <command>")
        print("Commands: upscale, benchmark, info")
