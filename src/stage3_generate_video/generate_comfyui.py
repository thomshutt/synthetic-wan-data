"""
Video Dataset Generator (ComfyUI) - Stage 3 of the synthetic data pipeline.

Converts image datasets to video using ComfyUI API with customizable workflows.

Usage:
    python -m src.stage3_generate_video.generate_comfyui \
        --input "./data/image_datasets/style_name" \
        --output "./data/video_datasets/style_name" \
        --workflow "./src/stage3_generate_video/video_wan2_2_14B_i2v.json" \
        --num-videos 10
"""

import argparse
import json
import random
import time
from pathlib import Path
from urllib.parse import urljoin

import requests
from PIL import Image


DEFAULT_COMFYUI_URL = "http://127.0.0.1:8188"
DEFAULT_WORKFLOW = Path(__file__).parent / "video_wan2_2_14B_i2v.json"

# Node IDs in the workflow template (update if using different workflow)
NODE_POSITIVE_PROMPT = "93"
NODE_LOAD_IMAGE = "116"
NODE_SAMPLER_SEED = "86"
NODE_SAVE_VIDEO = "108"
NODE_IMAGE_TO_VIDEO = "98"

# Default video settings
DEFAULT_MAX_PIXELS = 640 * 640  # ~409k pixels, adjust based on VRAM
DIMENSION_MULTIPLE = 32  # Dimensions must be divisible by this


def get_image_dimensions(image_path: Path) -> tuple[int, int]:
    """Get width and height of an image."""
    with Image.open(image_path) as img:
        return img.width, img.height


def calculate_video_dimensions(
    img_width: int,
    img_height: int,
    max_pixels: int = DEFAULT_MAX_PIXELS,
    multiple: int = DIMENSION_MULTIPLE,
) -> tuple[int, int]:
    """Calculate video dimensions preserving aspect ratio within pixel budget."""
    aspect_ratio = img_width / img_height

    # Calculate dimensions that fit within max_pixels while preserving aspect ratio
    # width * height = max_pixels, width/height = aspect_ratio
    # width = sqrt(max_pixels * aspect_ratio)
    # height = sqrt(max_pixels / aspect_ratio)
    import math
    width = math.sqrt(max_pixels * aspect_ratio)
    height = math.sqrt(max_pixels / aspect_ratio)

    # Round to nearest multiple
    width = int(round(width / multiple) * multiple)
    height = int(round(height / multiple) * multiple)

    # Ensure minimum dimensions
    width = max(width, multiple)
    height = max(height, multiple)

    return width, height

# Video file extensions to look for in outputs
VIDEO_EXTENSIONS = (".mp4", ".webm", ".gif")


def load_image_dataset(input_dir: Path) -> list[dict]:
    """Load image paths and captions from a stage 2 dataset."""
    items = []

    image_files = sorted(
        [f for f in input_dir.iterdir() if f.suffix.lower() in (".png", ".jpg", ".jpeg")]
    )

    for img_path in image_files:
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


def upload_image(comfyui_url: str, image_path: Path) -> str:
    """Upload an image to ComfyUI and return the filename."""
    url = urljoin(comfyui_url, "/upload/image")

    with open(image_path, "rb") as f:
        files = {"image": (image_path.name, f, "image/png")}
        response = requests.post(url, files=files)

    response.raise_for_status()
    result = response.json()
    return result["name"]


def queue_prompt(comfyui_url: str, workflow: dict) -> str:
    """Queue a prompt and return the prompt_id."""
    url = urljoin(comfyui_url, "/prompt")

    payload = {"prompt": workflow}
    response = requests.post(url, json=payload)
    response.raise_for_status()

    result = response.json()
    return result["prompt_id"]


def get_history(comfyui_url: str, prompt_id: str) -> dict | None:
    """Get history for a prompt_id. Returns None if not ready."""
    url = urljoin(comfyui_url, f"/history/{prompt_id}")

    response = requests.get(url)
    response.raise_for_status()

    history = response.json()
    if prompt_id in history:
        return history[prompt_id]
    return None


def wait_for_completion(
    comfyui_url: str,
    prompt_id: str,
    poll_interval: float = 2.0,
    timeout: float = 600.0,
) -> dict:
    """Poll until the prompt completes and return the history entry."""
    start_time = time.time()

    while True:
        elapsed = time.time() - start_time
        if elapsed > timeout:
            raise TimeoutError(f"Prompt {prompt_id} timed out after {timeout}s")

        history = get_history(comfyui_url, prompt_id)
        if history is not None:
            status = history.get("status", {})
            if status.get("completed", False):
                return history
            if status.get("status_str") == "error":
                raise RuntimeError(f"Prompt {prompt_id} failed: {status}")

        time.sleep(poll_interval)


def get_output_video_info(history: dict) -> dict | None:
    """Extract video output info from history for downloading."""
    outputs = history.get("outputs", {})

    # Look for video in various output keys
    for node_id, node_output in outputs.items():
        for key in ["videos", "gifs", "images"]:
            if key in node_output and isinstance(node_output[key], list):
                for file_info in node_output[key]:
                    if not isinstance(file_info, dict):
                        continue
                    filename = file_info.get("filename", "")
                    if filename.endswith(VIDEO_EXTENSIONS):
                        return {
                            "filename": filename,
                            "subfolder": file_info.get("subfolder", ""),
                            "type": file_info.get("type", "output"),
                        }
    return None


def download_video(comfyui_url: str, video_info: dict, output_path: Path) -> None:
    """Download a video from ComfyUI via the /view API."""
    params = {
        "filename": video_info["filename"],
        "subfolder": video_info["subfolder"],
        "type": video_info["type"],
    }
    url = urljoin(comfyui_url, "/view")

    response = requests.get(url, params=params, stream=True)
    response.raise_for_status()

    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)


def prepare_workflow(
    template: dict,
    uploaded_image_name: str,
    prompt: str,
    seed: int,
    output_prefix: str,
    width: int,
    height: int,
) -> dict:
    """Prepare workflow by injecting image, prompt, seed, output prefix, and dimensions."""
    workflow = json.loads(json.dumps(template))  # Deep copy

    # Inject positive prompt
    if NODE_POSITIVE_PROMPT in workflow:
        workflow[NODE_POSITIVE_PROMPT]["inputs"]["text"] = prompt

    # Inject uploaded image name
    if NODE_LOAD_IMAGE in workflow:
        workflow[NODE_LOAD_IMAGE]["inputs"]["image"] = uploaded_image_name

    # Inject seed
    if NODE_SAMPLER_SEED in workflow:
        workflow[NODE_SAMPLER_SEED]["inputs"]["noise_seed"] = seed

    # Inject output filename prefix
    if NODE_SAVE_VIDEO in workflow:
        workflow[NODE_SAVE_VIDEO]["inputs"]["filename_prefix"] = output_prefix

    # Inject video dimensions
    if NODE_IMAGE_TO_VIDEO in workflow:
        workflow[NODE_IMAGE_TO_VIDEO]["inputs"]["width"] = width
        workflow[NODE_IMAGE_TO_VIDEO]["inputs"]["height"] = height

    return workflow


def generate_video_dataset(
    input_dir: Path,
    output_dir: Path,
    workflow_path: Path,
    comfyui_url: str = DEFAULT_COMFYUI_URL,
    num_videos: int | None = None,
    seed: int | None = None,
    poll_interval: float = 2.0,
    timeout: float = 600.0,
    max_pixels: int = DEFAULT_MAX_PIXELS,
) -> dict:
    """Generate video dataset from images using ComfyUI."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(exist_ok=True)

    # Load workflow template
    print(f"\n[1/5] Loading workflow template from {workflow_path}...")
    with open(workflow_path, encoding="utf-8") as f:
        workflow_template = json.load(f)
    print(f"       Loaded workflow with {len(workflow_template)} nodes")

    # Load image dataset
    print(f"\n[2/5] Loading image dataset from {input_dir}...")
    items = load_image_dataset(input_dir)
    print(f"       Found {len(items)} images with captions")

    if not items:
        raise ValueError(f"No images found in {input_dir}")

    # Verify ComfyUI connection
    print(f"\n[3/5] Connecting to ComfyUI at {comfyui_url}...")
    try:
        response = requests.get(urljoin(comfyui_url, "/system_stats"))
        response.raise_for_status()
        print("       Connected successfully")
    except requests.RequestException as e:
        raise ConnectionError(f"Failed to connect to ComfyUI: {e}")

    # Initialize seed
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    print(f"       Base seed: {seed}")

    # Check for existing videos (for resume support)
    existing_videos = set()
    for f in output_dir.glob("*.mp4"):
        existing_videos.add(f.stem)

    # Generate videos
    target_count = num_videos if num_videos else len(items)
    print(f"\n[4/5] Generating {target_count} videos...")
    if existing_videos:
        print(f"       Found {len(existing_videos)} existing videos (will skip)")
    print()

    metadata = {
        "workflow": str(workflow_path),
        "comfyui_url": comfyui_url,
        "seed": seed,
        "source_dataset": str(input_dir),
        "videos": [],
    }

    generated_count = 0
    skipped_count = 0

    for i, item in enumerate(items):
        # Stop if we've generated enough
        if num_videos and generated_count >= num_videos:
            break

        output_name = f"{i:04d}"
        video_path = output_dir / f"{output_name}.mp4"
        caption_path = output_dir / f"{output_name}.txt"

        # Skip if already exists
        if output_name in existing_videos:
            print(f"  [skip] {item['name']} (already exists)")
            skipped_count += 1
            continue

        print(f"  [{generated_count+1}/{target_count}] Processing {item['name']}...")

        try:
            # Get image dimensions and calculate video size
            img_w, img_h = get_image_dimensions(item["image_path"])
            vid_w, vid_h = calculate_video_dimensions(img_w, img_h, max_pixels=max_pixels)
            print(f"           Image: {img_w}x{img_h} -> Video: {vid_w}x{vid_h}")

            # Upload image to ComfyUI
            print(f"           Uploading image...")
            uploaded_name = upload_image(comfyui_url, item["image_path"])

            # Prepare workflow with injected values
            output_prefix = f"synthetic/{output_name}"
            workflow = prepare_workflow(
                template=workflow_template,
                uploaded_image_name=uploaded_name,
                prompt=item["caption"],
                seed=seed + i,
                output_prefix=output_prefix,
                width=vid_w,
                height=vid_h,
            )

            # Queue prompt
            print(f"           Queuing prompt...")
            prompt_id = queue_prompt(comfyui_url, workflow)

            # Wait for completion
            print(f"           Generating...")
            history = wait_for_completion(
                comfyui_url,
                prompt_id,
                poll_interval=poll_interval,
                timeout=timeout,
            )

            # Get output video info and download via API
            video_info = get_output_video_info(history)
            if video_info:
                print(f"           Downloading...")
                download_video(comfyui_url, video_info, video_path)
                print(f"           Saved: {video_path.name}")
            else:
                print(f"           Warning: Could not find output video in history")
                continue

            # Save caption with trigger word for training
            caption_path.write_text(f"ddscope {item['caption']}", encoding="utf-8")

            metadata["videos"].append({
                "name": output_name,
                "source_image": item["name"],
                "caption": item["caption"],
                "prompt_id": prompt_id,
            })
            generated_count += 1

        except Exception as e:
            print(f"           Error: {e}")
            continue

    # Save metadata
    print(f"\n[5/5] Saving metadata...")
    metadata["generated"] = generated_count
    metadata["skipped"] = skipped_count
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
        description="Generate video dataset from images using ComfyUI"
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
        "--workflow",
        type=str,
        default=str(DEFAULT_WORKFLOW),
        help=f"ComfyUI workflow JSON file (default: {DEFAULT_WORKFLOW})",
    )
    parser.add_argument(
        "--comfyui-url",
        type=str,
        default=DEFAULT_COMFYUI_URL,
        help=f"ComfyUI API URL (default: {DEFAULT_COMFYUI_URL})",
    )
    parser.add_argument(
        "--num-videos",
        type=int,
        default=None,
        help="Number of videos to generate (default: all images)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: random)",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=2.0,
        help="Polling interval in seconds (default: 2.0)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="Timeout per video in seconds (default: 600.0)",
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=DEFAULT_MAX_PIXELS,
        help=f"Max pixels for video (width*height, default: {DEFAULT_MAX_PIXELS})",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Video Dataset Generator (ComfyUI)")
    print("=" * 60)

    metadata = generate_video_dataset(
        input_dir=Path(args.input),
        output_dir=Path(args.output),
        workflow_path=Path(args.workflow),
        comfyui_url=args.comfyui_url,
        num_videos=args.num_videos,
        seed=args.seed,
        poll_interval=args.poll_interval,
        timeout=args.timeout,
        max_pixels=args.max_pixels,
    )

    print(f"\n{'=' * 60}")
    print("Video generation complete!")
    print(f"{'=' * 60}")
    print(f"  Generated: {metadata['generated']} videos")
    if metadata.get('skipped', 0) > 0:
        print(f"  Skipped:   {metadata['skipped']} (already existed)")
    print(f"  Total:     {metadata['generated'] + metadata.get('skipped', 0)}")
    print(f"  Output: {args.output}")
    print()


if __name__ == "__main__":
    main()
