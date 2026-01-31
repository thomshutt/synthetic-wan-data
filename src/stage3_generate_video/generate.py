"""
Video Dataset Generator - Stage 3 of the synthetic data pipeline.

Converts image datasets to video using LTX-2 Image-to-Video.

Usage:
    python -m src.stage3_generate_video.generate \
        --input "./data/datasets/zombie" \
        --output "./data/video_datasets/zombie"
"""

import argparse
import json
from pathlib import Path

import torch
from diffusers import LTX2ImageToVideoPipeline
from diffusers.utils import load_image
from PIL import Image
from tqdm import tqdm


def load_image_dataset(input_dir: Path) -> list[dict]:
    """Load image paths and captions from a stage 2 dataset."""
    items = []

    # Find all images (png, jpg, jpeg)
    image_files = sorted(
        [f for f in input_dir.iterdir() if f.suffix.lower() in (".png", ".jpg", ".jpeg")]
    )

    for img_path in image_files:
        # Look for matching caption file
        caption_path = img_path.with_suffix(".txt")
        caption = ""
        if caption_path.exists():
            caption = caption_path.read_text(encoding="utf-8").strip()

        items.append({
            "image_path": img_path,
            "caption": caption,
            "name": img_path.stem,
        })

    return items


def save_video(frames: list, output_path: Path, fps: float):
    """Save PIL frames as MP4 video."""
    import imageio
    import numpy as np

    writer = imageio.get_writer(
        str(output_path),
        fps=fps,
        codec="libx264",
        quality=8,
    )

    for frame in frames:
        # Convert PIL to numpy if needed
        if hasattr(frame, "convert"):
            frame = np.array(frame)
        writer.append_data(frame)

    writer.close()


def generate_video_dataset(
    input_dir: Path,
    output_dir: Path,
    model: str = "Lightricks/LTX-2",
    num_frames: int = 121,
    frame_rate: float = 24.0,
    width: int = 768,
    height: int = 512,
    num_inference_steps: int = 40,
    guidance_scale: float = 4.0,
    negative_prompt: str = "worst quality, inconsistent motion, blurry, jittery, distorted, motion smear, motion artifacts",
    seed: int | None = None,
    max_videos: int | None = None,
    device: str = "cuda",
) -> dict:
    """Generate video dataset from images using LTX-2."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(exist_ok=True)

    # Load image dataset
    print(f"\n[1/4] Loading image dataset from {input_dir}...")
    items = load_image_dataset(input_dir)
    print(f"       Found {len(items)} images with captions")

    if max_videos:
        items = items[:max_videos]
        print(f"       Limited to {max_videos} videos (--max-videos)")

    if not items:
        raise ValueError(f"No images found in {input_dir}")

    # Load pipeline
    print(f"\n[2/4] Loading LTX-2 pipeline...")
    print(f"       Model: {model}")

    pipe = LTX2ImageToVideoPipeline.from_pretrained(
        model,
        torch_dtype=torch.bfloat16,
    )
    pipe.enable_model_cpu_offload(device=device)

    # Setup generator for reproducibility
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)
        print(f"       Seed: {seed}")

    # Generate videos
    print(f"\n[3/4] Generating {len(items)} videos...")
    print(f"       Resolution: {width}x{height}")
    print(f"       Frames: {num_frames} @ {frame_rate}fps")
    print()

    metadata = {
        "model": model,
        "num_frames": num_frames,
        "frame_rate": frame_rate,
        "width": width,
        "height": height,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "negative_prompt": negative_prompt,
        "seed": seed,
        "source_dataset": str(input_dir),
        "videos": [],
    }

    for i, item in enumerate(tqdm(items, desc="Generating videos")):
        output_name = f"{i:04d}"
        video_path = output_dir / f"{output_name}.mp4"
        caption_path = output_dir / f"{output_name}.txt"

        # Load and resize image
        image = load_image(str(item["image_path"]))
        image = image.resize((width, height), Image.Resampling.LANCZOS)

        # Generate video
        try:
            result = pipe(
                image=image,
                prompt=item["caption"],
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_frames=num_frames,
                frame_rate=frame_rate,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                output_type="pil",
                return_dict=True,
            )

            # Save video frames as mp4
            frames = result.frames[0]  # List of PIL images
            save_video(frames, video_path, frame_rate)

            # Save caption
            caption_path.write_text(item["caption"], encoding="utf-8")

            metadata["videos"].append({
                "name": output_name,
                "source_image": item["name"],
                "caption": item["caption"],
            })

        except Exception as e:
            print(f"\n       Error generating video for {item['name']}: {e}")
            continue

    # Save metadata
    print(f"\n[4/4] Saving metadata...")
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Generate dataset config for musubi-tuner
    config_path = output_dir / "dataset_config.toml"
    abs_output = str(output_dir.absolute()).replace("\\", "/")
    abs_cache = str(cache_dir.absolute()).replace("\\", "/")
    config_content = f'''# Dataset config for musubi-tuner (generated by stage3_generate_video)

[[datasets]]
video_directory = "{abs_output}"
caption_extension = ".txt"
num_repeats = 1
cache_directory = "{abs_cache}"
'''
    config_path.write_text(config_content, encoding="utf-8")

    print(f"       Metadata: {metadata_path}")
    print(f"       Config: {config_path}")

    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Generate video dataset from images using LTX-2"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input image dataset directory (from stage 2)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output video dataset directory",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Lightricks/LTX-2",
        help="LTX-2 model (default: Lightricks/LTX-2)",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=121,
        help="Frames per video, must be 8*N+1 (default: 121 = 5 seconds @ 24fps)",
    )
    parser.add_argument(
        "--frame-rate",
        type=float,
        default=24.0,
        help="Frame rate (default: 24.0)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=768,
        help="Video width, must be divisible by 32 (default: 768)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Video height, must be divisible by 32 (default: 512)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=40,
        help="Inference steps (default: 40)",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=4.0,
        help="Guidance scale (default: 4.0)",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default="worst quality, inconsistent motion, blurry, jittery, distorted, motion smear, motion artifacts",
        help="Negative prompt for quality control",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Limit number of videos (for testing)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on (default: cuda)",
    )

    args = parser.parse_args()

    print("=" * 50)
    print("Video Dataset Generator (LTX-2)")
    print("=" * 50)

    metadata = generate_video_dataset(
        input_dir=Path(args.input),
        output_dir=Path(args.output),
        model=args.model,
        num_frames=args.num_frames,
        frame_rate=args.frame_rate,
        width=args.width,
        height=args.height,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        max_videos=args.max_videos,
        device=args.device,
    )

    print(f"\n{'=' * 50}")
    print("Video generation complete!")
    print(f"{'=' * 50}")
    print(f"  Generated: {len(metadata['videos'])} videos")
    print(f"  Output: {args.output}")
    print()


if __name__ == "__main__":
    main()
