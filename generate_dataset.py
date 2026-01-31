"""
Simple Flux LoRA dataset generator for distilling styles into Wan2.1 LoRAs.

Usage:
    python generate_dataset.py --lora path/to/lora.safetensors --prompts prompts.txt --output ./output
"""

import argparse
import json
from pathlib import Path

import torch
from diffusers import FluxPipeline
from tqdm import tqdm


def load_prompts(prompts_file: Path) -> list[str]:
    """Load prompts from file."""
    print(f"[1/4] Loading prompts from {prompts_file}...")
    with open(prompts_file) as f:
        prompts = [line.strip() for line in f if line.strip()]
    print(f"       Loaded {len(prompts)} prompts")
    return prompts


def generate_dataset(
    flux_path: str,
    lora_path: str,
    output_dir: Path,
    prompts: list[str],
    num_images: int = 50,
    lora_scale: float = 1.0,
    style_prefix: str = "",
    style_suffix: str = "",
    resolution: tuple[int, int] = (1024, 1024),
    guidance_scale: float = 3.5,
    num_steps: int = 28,
    seed: int | None = None,
):
    """Generate images using Flux + LoRA and save with captions."""

    print(f"\n[2/4] Loading Flux pipeline...")
    print(f"       Model: {flux_path}")

    if flux_path.startswith("black-forest-labs/") or flux_path.startswith("hf:"):
        # Load from HuggingFace
        hf_id = flux_path.replace("hf:", "")
        print(f"       Loading from HuggingFace: {hf_id}")
        pipe = FluxPipeline.from_pretrained(
            hf_id,
            torch_dtype=torch.bfloat16,
        )
    else:
        # Load from local file
        print(f"       Loading from local file...")
        pipe = FluxPipeline.from_single_file(
            flux_path,
            torch_dtype=torch.bfloat16,
        )

    pipe.enable_model_cpu_offload()
    print(f"       Flux loaded successfully")

    print(f"\n[3/4] Loading LoRA...")
    print(f"       LoRA: {lora_path}")
    print(f"       Scale: {lora_scale}")
    lora_name = Path(lora_path).stem
    pipe.load_lora_weights(lora_path, adapter_name=lora_name)
    pipe.set_adapters([lora_name], adapter_weights=[lora_scale])
    print(f"       LoRA '{lora_name}' loaded successfully")

    print(f"\n[4/4] Generating {num_images} images...")
    print(f"       Output: {output_dir}")
    print(f"       Resolution: {resolution[0]}x{resolution[1]}")
    print(f"       Steps: {num_steps}, Guidance: {guidance_scale}")
    if style_prefix:
        print(f"       Prefix: '{style_prefix}'")
    if style_suffix:
        print(f"       Suffix: '{style_suffix}'")
    if seed is not None:
        print(f"       Seed: {seed}")
    print()

    output_dir.mkdir(parents=True, exist_ok=True)

    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)

    for i in tqdm(range(num_images), desc="Generating"):
        # Pick a prompt (cycle through if needed)
        base_prompt = prompts[i % len(prompts)]

        # Add style prefix/suffix
        full_prompt = f"{style_prefix} {base_prompt} {style_suffix}".strip()

        # Generate
        if seed is not None:
            generator.manual_seed(seed + i)

        image = pipe(
            prompt=full_prompt,
            width=resolution[0],
            height=resolution[1],
            guidance_scale=guidance_scale,
            num_inference_steps=num_steps,
            generator=generator,
        ).images[0]

        # Save image
        image_path = output_dir / f"{i:04d}.png"
        image.save(image_path)

        # Save caption
        caption_path = output_dir / f"{i:04d}.txt"
        caption_path.write_text(full_prompt)

        tqdm.write(f"  [{i+1}/{num_images}] Saved {image_path.name}")

    # Save metadata
    print(f"\nSaving metadata...")
    metadata = {
        "flux_path": flux_path,
        "lora_path": lora_path,
        "lora_scale": lora_scale,
        "style_prefix": style_prefix,
        "style_suffix": style_suffix,
        "resolution": resolution,
        "guidance_scale": guidance_scale,
        "num_steps": num_steps,
        "num_images": num_images,
        "seed": seed,
    }
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved {metadata_path}")

    # Generate musubi-tuner dataset config
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(exist_ok=True)

    # Use forward slashes for TOML compatibility
    img_dir = str(output_dir.absolute()).replace("\\", "/")
    cache_dir_str = str(cache_dir.absolute()).replace("\\", "/")

    dataset_config = f'''# Auto-generated musubi-tuner dataset config
# LoRA source: {lora_path}

[general]
resolution = [{resolution[0]}, {resolution[1]}]
caption_extension = ".txt"
enable_bucket = true
bucket_no_upscale = true

[[datasets]]
image_directory = "{img_dir}"
cache_directory = "{cache_dir_str}"
num_repeats = 1
'''
    config_path = output_dir / "dataset_config.toml"
    config_path.write_text(dataset_config)
    print(f"  Saved {config_path}")

    print(f"\n{'='*50}")
    print(f"Done! Generated {num_images} images")
    print(f"{'='*50}")
    print(f"  Images:  {output_dir}/*.png")
    print(f"  Captions: {output_dir}/*.txt")
    print(f"  Config:   {config_path}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Generate dataset from Flux + LoRA")
    parser.add_argument("--flux", type=str, default="black-forest-labs/FLUX.1-dev", help="Flux model path or HuggingFace ID")
    parser.add_argument("--lora", type=str, required=True, help="Path to LoRA safetensors")
    parser.add_argument("--prompts", type=str, required=True, help="Path to prompts file (one per line)")
    parser.add_argument("--output", type=str, default="./dataset", help="Output directory")
    parser.add_argument("--num_images", type=int, default=None, help="Number of images to generate (default: number of prompts)")
    parser.add_argument("--lora_scale", type=float, default=1.0, help="LoRA scale/strength")
    parser.add_argument("--style_prefix", type=str, default="", help="Prefix to add to all prompts")
    parser.add_argument("--style_suffix", type=str, default="", help="Suffix to add to all prompts")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--num_steps", type=int, default=28)
    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()

    print("=" * 50)
    print("Flux LoRA Dataset Generator")
    print("=" * 50)

    prompts = load_prompts(Path(args.prompts))

    # Default to number of prompts if not specified
    num_images = args.num_images if args.num_images is not None else len(prompts)

    generate_dataset(
        flux_path=args.flux,
        lora_path=args.lora,
        output_dir=Path(args.output),
        prompts=prompts,
        num_images=num_images,
        lora_scale=args.lora_scale,
        style_prefix=args.style_prefix,
        style_suffix=args.style_suffix,
        resolution=(args.width, args.height),
        guidance_scale=args.guidance_scale,
        num_steps=args.num_steps,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
