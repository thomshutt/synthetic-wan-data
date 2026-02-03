"""
Preprocess video clips for LoRA training.

Splits long videos into training-sized chunks and generates captions.
"""

import argparse
import subprocess
import json
from pathlib import Path


def get_video_info(video_path: Path) -> dict:
    """Get video metadata using ffprobe."""
    result = subprocess.run([
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", "-show_streams", str(video_path)
    ], capture_output=True, text=True)
    return json.loads(result.stdout)


def split_video(
    input_path: Path,
    output_dir: Path,
    segment_duration: float = 2.5,
    output_fps: int = 24,
    prefix: str = "",
) -> list[Path]:
    """
    Split video into segments without introducing static frames.

    Uses re-encoding to ensure clean cuts at any point.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get video info
    info = get_video_info(input_path)
    stream = next(s for s in info["streams"] if s["codec_type"] == "video")
    duration = float(info["format"]["duration"])

    print(f"  Input: {input_path.name}")
    print(f"  Duration: {duration:.1f}s, splitting into {segment_duration}s segments")

    output_files = []
    segment_idx = 0
    start_time = 0.0

    while start_time < duration - 0.5:  # Don't create tiny segments at end
        output_name = f"{prefix}{segment_idx:03d}.mp4"
        output_path = output_dir / output_name

        # Use ffmpeg to extract and re-encode segment
        # -ss before -i for fast seeking, re-encode for clean cuts
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_time),
            "-i", str(input_path),
            "-t", str(segment_duration),
            "-r", str(output_fps),  # Output framerate
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",
            "-an",  # No audio
            "-pix_fmt", "yuv420p",
            str(output_path)
        ]

        subprocess.run(cmd, capture_output=True)

        if output_path.exists() and output_path.stat().st_size > 1000:
            output_files.append(output_path)
            print(f"    Created: {output_name}")

        segment_idx += 1
        start_time += segment_duration

    return output_files


def generate_caption(video_name: str, style_description: str) -> str:
    """Generate caption for a video clip based on the style."""
    return f"{style_description} ddscope"


def preprocess_videos(
    input_dir: Path,
    output_dir: Path,
    segment_duration: float = 2.5,
    output_fps: int = 24,
    captions: dict[str, str] = None,
):
    """
    Process all videos in input directory.

    Args:
        input_dir: Directory containing source videos
        output_dir: Output directory for clips and captions
        segment_duration: Length of each clip in seconds
        output_fps: Output framerate (24 for Wan2.1)
        captions: Dict mapping video filename patterns to captions
    """
    clips_dir = output_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    # Find all video files
    video_files = list(input_dir.glob("*.mp4")) + list(input_dir.glob("*.MP4"))

    print(f"Found {len(video_files)} video(s) to process")
    print(f"Segment duration: {segment_duration}s, Output FPS: {output_fps}")
    print()

    all_clips = []

    for video_path in sorted(video_files):
        # Create prefix from source filename
        prefix = video_path.stem + "_"

        clips = split_video(
            video_path,
            clips_dir,
            segment_duration=segment_duration,
            output_fps=output_fps,
            prefix=prefix,
        )

        # Generate captions for clips
        caption = captions.get(video_path.stem, "") if captions else ""
        for clip_path in clips:
            caption_path = clip_path.with_suffix(".txt")
            caption_path.write_text(caption, encoding="utf-8")

        all_clips.extend(clips)
        print()

    # Generate dataset config
    config_path = output_dir / "dataset_config.toml"
    clips_abs = str(clips_dir.absolute()).replace("\\", "/")
    cache_dir = str((output_dir / "cache").absolute()).replace("\\", "/")

    config = f'''# Video LoRA dataset config
# Source: {input_dir}

[general]
resolution = [832, 480]
caption_extension = ".txt"
enable_bucket = true
bucket_no_upscale = true

[[datasets]]
video_directory = "{clips_abs}"
cache_directory = "{cache_dir}"
num_repeats = 1
'''
    config_path.write_text(config)

    print("=" * 50)
    print(f"Preprocessing complete!")
    print(f"  Clips: {len(all_clips)}")
    print(f"  Output: {clips_dir}")
    print(f"  Config: {config_path}")
    print("=" * 50)

    return all_clips


def main():
    parser = argparse.ArgumentParser(description="Preprocess videos for LoRA training")
    parser.add_argument("input_dir", type=str, help="Input directory with videos")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument("--segment-duration", type=float, default=2.5, help="Segment length in seconds")
    parser.add_argument("--fps", type=int, default=24, help="Output framerate")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output) if args.output else input_dir.parent / input_dir.name.replace("staging/", "")

    preprocess_videos(
        input_dir=input_dir,
        output_dir=output_dir,
        segment_duration=args.segment_duration,
        output_fps=args.fps,
    )


if __name__ == "__main__":
    main()
