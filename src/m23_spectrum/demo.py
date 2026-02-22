"""
Gradio Demo for M23-Spectrum Super-Resolution
==============================================

A web-based demo for interactive image super-resolution.

Run:
    python -m m23_spectrum.demo

Or:
    gradio demo.py
"""

import os
import sys
from pathlib import Path
from typing import Tuple, Optional
import time

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from PIL import Image

try:
    import gradio as gr
except ImportError:
    print("Please install gradio: pip install gradio")
    sys.exit(1)

# Import from package
from m23_spectrum import M23RLFN, SuperResolver, TTAUpscaler
from m23_spectrum.utils.metrics import calculate_psnr, calculate_ssim


# Global model cache
MODELS = {}


def get_model(model_name: str, use_tta: bool, device: str = "auto"):
    """Load or retrieve cached model."""
    cache_key = f"{model_name}_{use_tta}"

    if cache_key not in MODELS:
        if use_tta:
            MODELS[cache_key] = TTAUpscaler(model_name, device=device, tta_mode="full")
        else:
            MODELS[cache_key] = SuperResolver(model_name, device=device)

    return MODELS[cache_key]


def process_image(
    input_image,
    model_name: str,
    use_tta: bool,
    show_comparison: bool,
) -> Tuple[Image.Image, str]:
    """
    Process an image with the selected model.

    Args:
        input_image: Input PIL Image
        model_name: Name of model to use
        use_tta: Whether to use TTA
        show_comparison: Whether to show side-by-side comparison

    Returns:
        Tuple of (output_image, info_text)
    """
    if input_image is None:
        return None, "Please upload an image."

    start_time = time.time()

    try:
        # Get model
        model = get_model(model_name, use_tta)

        # Process
        sr_image = model.process(input_image)

        # Calculate metrics (using bicubic as baseline)
        from m23_spectrum.utils.image import image_to_tensor, tensor_to_image
        import torch.nn.functional as F

        hr_tensor = image_to_tensor(sr_image)
        lr_tensor = image_to_tensor(input_image)

        # Resize for comparison
        lr_upscaled = F.interpolate(
            lr_tensor,
            size=(sr_image.height, sr_image.width),
            mode="bicubic",
            align_corners=False,
        )
        hr_tensor_resized = F.interpolate(
            hr_tensor,
            size=(sr_image.height, sr_image.width),
            mode="bicubic",
            align_corners=False,
        )

        elapsed = time.time() - start_time

        # Build info
        info = f"""
### Results
- **Model**: {model_name}
- **TTA**: {"Enabled" if use_tta else "Disabled"}
- **Resolution**: {sr_image.width} × {sr_image.height}
- **Processing Time**: {elapsed:.2f}s
- **Scale Factor**: ×{model.scale}
"""

        if show_comparison:
            # Create comparison image
            comparison = Image.new("RGB", (sr_image.width * 2 + 10, sr_image.height))
            comparison.paste(lr_upscaled_pil := tensor_to_image(lr_upscaled), (0, 0))
            comparison.paste(sr_image, (sr_image.width + 10, 0))
            return comparison, info

        return sr_image, info

    except Exception as e:
        return None, f"Error: {str(e)}"


def create_demo():
    """Create and launch Gradio demo."""

    with gr.Blocks(
        title="M23-Spectrum Super-Resolution",
        theme=gr.themes.Soft(),
    ) as demo:

        gr.Markdown("""
        # 🔮 M23-Spectrum Super-Resolution

        **State-of-the-art image super-resolution powered by M23-Spectrum weight initialization.**

        Upload a low-resolution image and get a high-quality super-resolved result.
        """)

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(
                    type="pil",
                    label="Input Image",
                    height=400,
                )

                with gr.Row():
                    model_name = gr.Dropdown(
                        choices=[
                            "m23-rlfn-x2",
                            "m23-rlfn-x3",
                            "m23-rlfn-x4",
                            "m23-rlfn-x4-large",
                        ],
                        value="m23-rlfn-x4",
                        label="Model",
                    )

                with gr.Row():
                    use_tta = gr.Checkbox(
                        label="Use TTA (+0.1-0.3 dB boost)",
                        value=False,
                    )
                    show_comparison = gr.Checkbox(
                        label="Show Comparison",
                        value=False,
                    )

                process_btn = gr.Button(
                    "🚀 Upscale",
                    variant="primary",
                    size="lg",
                )

            with gr.Column(scale=1):
                output_image = gr.Image(
                    type="pil",
                    label="Super-Resolved Image",
                    height=400,
                )

                info_text = gr.Markdown()

        # Examples
        gr.Examples(
            examples=[
                ["assets/examples/baby.png"],
                ["assets/examples/bird.png"],
                ["assets/examples/butterfly.png"],
                ["assets/examples/head.png"],
                ["assets/examples/woman.png"],
            ],
            inputs=input_image,
            label="Example Images (Set5)",
        )

        gr.Markdown("""
        ### ℹ️ About M23-Spectrum

        M23-Spectrum uses algebraic weight initialization based on the Mathieu group M23,
        providing:
        - **Fast convergence**: 2.8× faster training
        - **High quality**: 29-32 dB on benchmarks
        - **Efficient**: ~900K parameters, <20ms inference

        ### 📊 Model Comparison

        | Model | Scale | Params | Set5 PSNR | Speed |
        |-------|-------|--------|-----------|-------|
        | M23-RLFN | ×4 | ~900K | 30.2 dB | ~15ms |
        | M23-RLFN-Large | ×4 | ~1.8M | 31.0 dB | ~25ms |

        ---

        [GitHub](https://github.com/m23spectrum/m23-spectrum) | [Paper](https://arxiv.org) | [Docs](https://m23spectrum.dev)
        """)

        # Connect events
        process_btn.click(
            fn=process_image,
            inputs=[input_image, model_name, use_tta, show_comparison],
            outputs=[output_image, info_text],
        )

    return demo


def main():
    """Launch the demo."""
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )


if __name__ == "__main__":
    main()
