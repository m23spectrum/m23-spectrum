"""Command-line interface for M23-Spectrum."""

import argparse
import sys
from pathlib import Path


def main_upscale():
    """CLI entry point for upscaling."""
    parser = argparse.ArgumentParser(
        description="M23-Spectrum Image Super-Resolution",
    )

    parser.add_argument("input", type=str, help="Input image path or directory")
    parser.add_argument("output", type=str, help="Output image path or directory")
    parser.add_argument("--model", type=str, default="m23-rlfn-x4",
                        choices=["m23-rlfn-x2", "m23-rlfn-x3", "m23-rlfn-x4",
                                 "m23-rlfn-x4-large"])
    parser.add_argument("--tta", action="store_true", help="Use TTA")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--half", action="store_true", help="Use FP16")

    args = parser.parse_args()

    from .inference import SuperResolver, TTAUpscaler

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: Input not found: {input_path}")
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.model}")
    if args.tta:
        upscaler = TTAUpscaler(args.model, device=args.device)
    else:
        upscaler = SuperResolver(args.model, device=args.device, half=args.half)

    if input_path.is_file():
        print(f"Processing: {input_path}")
        upscaler.upscale(input_path, output_path)
        print(f"Saved: {output_path}")
    else:
        output_path.mkdir(parents=True, exist_ok=True)
        image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
        images = [f for f in input_path.iterdir()
                  if f.suffix.lower() in image_extensions]

        for i, img_path in enumerate(images, 1):
            out_path = output_path / img_path.name
            print(f"[{i}/{len(images)}] Processing: {img_path.name}")
            upscaler.upscale(img_path, out_path)

        print(f"Done! Results saved to: {output_path}")


def main_benchmark():
    """CLI entry point for benchmarking."""
    parser = argparse.ArgumentParser(description="M23-Spectrum Benchmark")

    parser.add_argument("--dataset", type=str, default="set5",
                        choices=["set5", "set14", "bsd100", "urban100", "all"])
    parser.add_argument("--data-dir", type=str, default="data/benchmark")
    parser.add_argument("--model", type=str, default="m23-rlfn-x4")
    parser.add_argument("--scale", type=int, default=4, choices=[2, 3, 4])
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--save-dir", type=str, default=None)

    args = parser.parse_args()

    from . import SuperResolver, TTAUpscaler
    from .data.benchmark import (
        Set5Dataset, Set14Dataset, BSD100Dataset, evaluate_model,
    )

    print(f"Loading model: {args.model}")
    if args.tta:
        upscaler = TTAUpscaler(args.model, device="auto")
        model = upscaler.model
    else:
        upscaler = SuperResolver(args.model, device="auto")
        model = upscaler.model

    data_dir = Path(args.data_dir)

    datasets_map = {
        "set5": [("Set5", Set5Dataset)],
        "set14": [("Set14", Set14Dataset)],
        "bsd100": [("BSD100", BSD100Dataset)],
        "all": [("Set5", Set5Dataset), ("Set14", Set14Dataset),
                ("BSD100", BSD100Dataset)],
    }

    print("\n" + "=" * 60)
    print("  M23-Spectrum Benchmark Results")
    print("=" * 60)

    results = []
    for name, dataset_class in datasets_map.get(args.dataset, []):
        dataset_path = data_dir / name
        if not dataset_path.exists():
            continue

        dataset = dataset_class(str(dataset_path), scale=args.scale)
        result = evaluate_model(model, dataset, save_dir=args.save_dir)
        results.append(result)
        print(f"  {name}: PSNR = {result['psnr_mean']:.2f} dB | "
              f"SSIM = {result['ssim_mean']:.4f}")

    if results:
        avg_psnr = sum(r['psnr_mean'] for r in results) / len(results)
        avg_ssim = sum(r['ssim_mean'] for r in results) / len(results)
        print("-" * 60)
        print(f"  Average: PSNR = {avg_psnr:.2f} dB | SSIM = {avg_ssim:.4f}")


def main_info():
    """Print model and device information."""
    from .models.factory import print_model_table
    from .utils.device import print_device_info

    print()
    print_model_table()
    print()
    print_device_info()
