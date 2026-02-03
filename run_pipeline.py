"""
Unified pipeline for Flux-to-Wan LoRA distillation.

Runs any combination of stages for one or more models:
  Stage 1: Scrape LoRA + expand prompts (CivitAI)
  Stage 2: Generate images (Flux + LoRA)
  Stage 3: Generate videos (ComfyUI I2V)

Usage:
    # Single model, all stages
    uv run python run_pipeline.py --model-id 245889 --stages 1,2,3

    # Single model, specific stages
    uv run python run_pipeline.py --model-id 245889 --stages 2,3

    # Multiple models
    uv run python run_pipeline.py --model-id 245889 179353 180780 --stages 1,2

    # Batch from file
    uv run python run_pipeline.py --models-file models.txt --stages 1,2,3

    # Just create dataset config
    uv run python run_pipeline.py --model-id 245889 --create-config

    # Output training commands
    uv run python run_pipeline.py --model-id 245889 --training-commands
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


# Default paths (adjust for your system)
DATA_DIR = Path("./data")
VAE_PATH = "C:/Users/ryanf/.daydream-scope/models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"
T5_PATH = "C:/Users/ryanf/.daydream-scope/models/WanVideo_comfy/umt5-xxl-enc-fp8_e4m3fn.safetensors"
DIT_PATH = "C:/Users/ryanf/.daydream-scope/models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors"

# Default generation settings
DEFAULT_NUM_IMAGES = 30
DEFAULT_NUM_VIDEOS = 20
DEFAULT_OLLAMA_MODEL = "llama3.2:3b"

# Default training settings
DEFAULT_EPOCHS = 32
DEFAULT_SAVE_EVERY = 8
DEFAULT_IMAGE_RESOLUTION = [768, 768]
DEFAULT_VIDEO_RESOLUTION = [640, 640]
DEFAULT_TARGET_FRAMES = [33]


def run_stage1(model_id: int, num_prompts: int = 50, ollama_model: str = DEFAULT_OLLAMA_MODEL) -> None:
    """Stage 1: Scrape LoRA and expand prompts."""
    url = f"https://civitai.com/models/{model_id}"
    print(f"\n{'='*60}")
    print(f"STAGE 1: Scraping {url}")
    print(f"{'='*60}")

    cmd = [
        sys.executable, "-m", "src.stage1_scrape.scraper",
        url,
        "--data-dir", str(DATA_DIR),
        "--expand-to", str(num_prompts),
        "--ollama-model", ollama_model,
    ]
    subprocess.run(cmd, check=True)


def run_stage2(model_id: int, num_images: int = DEFAULT_NUM_IMAGES) -> None:
    """Stage 2: Generate images with Flux + LoRA."""
    model_dir = DATA_DIR / f"model_{model_id}"
    prompts_json = model_dir / "prompts.json"
    prompts_txt = model_dir / "prompts.txt"
    image_dir = model_dir / "images"

    if not prompts_json.exists():
        raise FileNotFoundError(f"Prompts JSON not found: {prompts_json}\nRun stage 1 first.")

    with open(prompts_json, encoding="utf-8") as f:
        data = json.load(f)

    lora_path = data.get("lora_path")
    if not lora_path:
        raise ValueError(f"No lora_path in {prompts_json}")

    if not Path(lora_path).exists():
        raise FileNotFoundError(f"LoRA not found: {lora_path}")

    print(f"\n{'='*60}")
    print(f"STAGE 2: Generating {num_images} images for model {model_id}")
    print(f"{'='*60}")

    cmd = [
        sys.executable, "-m", "src.stage2_generate_images.generate",
        "--lora", lora_path,
        "--prompts", str(prompts_txt),
        "--output", str(image_dir),
        "--num_images", str(num_images),
    ]
    subprocess.run(cmd, check=True)


def run_stage3(model_id: int, num_videos: int = DEFAULT_NUM_VIDEOS) -> None:
    """Stage 3: Generate videos with ComfyUI."""
    model_dir = DATA_DIR / f"model_{model_id}"
    image_dir = model_dir / "images"
    video_dir = model_dir / "videos"

    if not image_dir.exists():
        raise FileNotFoundError(f"Images not found: {image_dir}\nRun stage 2 first.")

    print(f"\n{'='*60}")
    print(f"STAGE 3: Generating {num_videos} videos for model {model_id}")
    print(f"  Make sure ComfyUI is running!")
    print(f"{'='*60}")

    cmd = [
        sys.executable, "-m", "src.stage3_generate_video.generate_comfyui",
        "--input", str(image_dir),
        "--output", str(video_dir),
        "--num-videos", str(num_videos),
    ]
    subprocess.run(cmd, check=True)


def create_dataset_config(
    model_id: int,
    image_resolution: list[int] = DEFAULT_IMAGE_RESOLUTION,
    video_resolution: list[int] = DEFAULT_VIDEO_RESOLUTION,
    target_frames: list[int] = DEFAULT_TARGET_FRAMES,
    image_repeats: int = 2,
    video_repeats: int = 1,
) -> Path:
    """Create dataset_config.toml for training."""
    model_dir = DATA_DIR / f"model_{model_id}"
    images_dir = model_dir / "images"
    videos_dir = model_dir / "videos"
    cache_dir = model_dir / "cache"
    config_path = model_dir / "dataset_config.toml"

    abs_images = str(images_dir.resolve()).replace("\\", "/")
    abs_videos = str(videos_dir.resolve()).replace("\\", "/")
    abs_cache_images = str((cache_dir / "images").resolve()).replace("\\", "/")
    abs_cache_videos = str((cache_dir / "videos").resolve()).replace("\\", "/")

    has_images = images_dir.exists() and any(images_dir.glob("*.png")) or any(images_dir.glob("*.jpg"))
    has_videos = videos_dir.exists() and any(videos_dir.glob("*.mp4"))

    config_lines = [
        "# Dataset config for musubi-tuner",
        f"# Model: {model_id}",
        "",
        "[general]",
        'caption_extension = ".txt"',
        "batch_size = 1",
        "enable_bucket = true",
        "bucket_no_upscale = true",
    ]

    if has_images:
        config_lines.extend([
            "",
            f"# Image dataset ({len(list(images_dir.glob('*.png')) + list(images_dir.glob('*.jpg')))} images)",
            "[[datasets]]",
            f'image_directory = "{abs_images}"',
            f'cache_directory = "{abs_cache_images}"',
            f"resolution = {image_resolution}",
            f"num_repeats = {image_repeats}",
        ])

    if has_videos:
        config_lines.extend([
            "",
            f"# Video dataset ({len(list(videos_dir.glob('*.mp4')))} videos)",
            "# IMPORTANT: target_frames is required! Default is [1] which only uses 1 frame!",
            "[[datasets]]",
            f'video_directory = "{abs_videos}"',
            f'cache_directory = "{abs_cache_videos}"',
            f"resolution = {video_resolution}",
            f"target_frames = {target_frames}",
            'frame_extraction = "head"',
            f"num_repeats = {video_repeats}",
        ])

    if not has_images and not has_videos:
        print(f"  Warning: No images or videos found for model {model_id}")
        return None

    config_path.write_text("\n".join(config_lines) + "\n", encoding="utf-8")
    print(f"  Created: {config_path}")
    return config_path


def generate_cache_commands(model_id: int) -> list[str]:
    """Generate caching commands for a model."""
    config_path = f"../data/model_{model_id}/dataset_config.toml"
    return [
        f'uv run python src/musubi_tuner/wan_cache_latents.py --dataset_config "{config_path}" --vae "{VAE_PATH}"',
        f'uv run python src/musubi_tuner/wan_cache_text_encoder_outputs.py --dataset_config "{config_path}" --t5 "{T5_PATH}"',
    ]


def generate_train_command(
    model_id: int,
    output_name: str = None,
    epochs: int = DEFAULT_EPOCHS,
    save_every: int = DEFAULT_SAVE_EVERY,
) -> str:
    """Generate training command for a model."""
    config_path = f"../data/model_{model_id}/dataset_config.toml"
    if output_name is None:
        output_name = f"model_{model_id}"

    return (
        f'uv run accelerate launch --mixed_precision bf16 src/musubi_tuner/wan_train_network.py '
        f'--task t2v-1.3B --dit "{DIT_PATH}" '
        f'--dataset_config "{config_path}" '
        f'--output_dir "./output/model_{model_id}" --output_name {output_name} '
        f'--sdpa --network_module networks.lora_wan --network_dim 64 '
        f'--optimizer_type adamw8bit --learning_rate 2e-4 '
        f'--gradient_checkpointing --timestep_sampling shift --discrete_flow_shift 3.0 '
        f'--max_train_epochs {epochs} --save_every_n_epochs {save_every} '
        f'--save_state --seed 42'
    )


def get_model_name(model_id: int) -> str:
    """Try to get a friendly name from prompts.json."""
    prompts_json = DATA_DIR / f"model_{model_id}" / "prompts.json"
    if prompts_json.exists():
        try:
            with open(prompts_json, encoding="utf-8") as f:
                data = json.load(f)
            # Try to extract name from lora filename
            lora_path = data.get("lora_path", "")
            if lora_path:
                name = Path(lora_path).stem
                # Clean up common patterns
                name = name.replace("-flux", "").replace("_flux", "")
                name = name.replace("-", "_").lower()
                return name
        except Exception:
            pass
    return f"model_{model_id}"


def parse_stages(stages_str: str) -> list[int]:
    """Parse stages string like '1,2,3' into list of ints."""
    return [int(s.strip()) for s in stages_str.split(",")]


def load_models_from_file(filepath: Path) -> list[int]:
    """Load model IDs from a file (one per line)."""
    model_ids = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                model_ids.append(int(line))
    return model_ids


def main():
    parser = argparse.ArgumentParser(
        description="Unified pipeline for Flux-to-Wan LoRA distillation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline for one model
  python run_pipeline.py --model-id 245889 --stages 1,2,3

  # Just images and videos (scraping already done)
  python run_pipeline.py --model-id 245889 --stages 2,3

  # Multiple models, images only
  python run_pipeline.py --model-id 245889 179353 --stages 1,2

  # Create dataset config for training
  python run_pipeline.py --model-id 245889 --create-config

  # Output training commands
  python run_pipeline.py --model-id 245889 --training-commands
        """,
    )

    # Model selection
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument(
        "--model-id", type=int, nargs="+",
        help="Model ID(s) to process",
    )
    model_group.add_argument(
        "--models-file", type=Path,
        help="File containing model IDs (one per line)",
    )

    # Stage selection
    parser.add_argument(
        "--stages", type=str, default="1,2,3",
        help="Stages to run (comma-separated, e.g., '1,2,3' or '2,3')",
    )

    # Generation settings
    parser.add_argument("--num-images", type=int, default=DEFAULT_NUM_IMAGES, help="Images to generate")
    parser.add_argument("--num-videos", type=int, default=DEFAULT_NUM_VIDEOS, help="Videos to generate")
    parser.add_argument("--num-prompts", type=int, default=50, help="Prompts to expand to")
    parser.add_argument("--ollama-model", type=str, default=DEFAULT_OLLAMA_MODEL, help="Ollama model for expansion")

    # Config/command generation (mutually exclusive with running stages)
    parser.add_argument(
        "--create-config", action="store_true",
        help="Create dataset_config.toml for training (doesn't run stages)",
    )
    parser.add_argument(
        "--training-commands", action="store_true",
        help="Output training commands (doesn't run stages)",
    )

    # Config settings
    parser.add_argument("--target-frames", type=int, default=33, help="Target frames for video training")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Training epochs")

    args = parser.parse_args()

    # Get model IDs
    if args.models_file:
        model_ids = load_models_from_file(args.models_file)
    elif args.model_id:
        model_ids = args.model_id
    else:
        parser.error("Must specify --model-id or --models-file")

    stages = parse_stages(args.stages)

    # Handle config creation
    if args.create_config:
        print(f"Creating dataset configs for {len(model_ids)} model(s)...")
        for model_id in model_ids:
            create_dataset_config(
                model_id,
                target_frames=[args.target_frames],
            )
        return

    # Handle training command generation
    if args.training_commands:
        print("# Training commands (run from musubi-tuner directory)")
        print("# " + "=" * 58)
        print()

        for model_id in model_ids:
            name = get_model_name(model_id)
            print(f"# === model_{model_id} ({name}) ===")
            print(f"# Caching:")
            for cmd in generate_cache_commands(model_id):
                print(cmd)
            print()
            print(f"# Training:")
            print(generate_train_command(model_id, name, args.epochs))
            print()
        return

    # Run stages
    print("=" * 60)
    print(f"PIPELINE: {len(model_ids)} model(s), stages {stages}")
    print(f"  Images: {args.num_images}, Videos: {args.num_videos}")
    print("=" * 60)

    for i, model_id in enumerate(model_ids):
        print(f"\n\n{'#'*60}")
        print(f"# MODEL {i+1}/{len(model_ids)}: {model_id}")
        print(f"{'#'*60}")

        try:
            if 1 in stages:
                run_stage1(model_id, args.num_prompts, args.ollama_model)

            if 2 in stages:
                run_stage2(model_id, args.num_images)

            if 3 in stages:
                run_stage3(model_id, args.num_videos)

            print(f"\n  Completed model {model_id}")

        except Exception as e:
            print(f"\n  FAILED: {e}")
            continue

    print(f"\n\n{'='*60}")
    print("PIPELINE COMPLETE")
    print(f"{'='*60}")
    print("\nNext steps:")
    print(f"  1. Create configs: python run_pipeline.py --model-id {' '.join(str(m) for m in model_ids)} --create-config")
    print(f"  2. Get training commands: python run_pipeline.py --model-id {' '.join(str(m) for m in model_ids)} --training-commands")


if __name__ == "__main__":
    main()
