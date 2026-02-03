# Distill Flux LoRAs to Wan2.1 LoRAs

Four-stage pipeline for distilling Flux LoRA styles into Wan2.1 video LoRAs.

```
Stage 1: Scrape   -> Stage 2: Images -> Stage 3: Videos -> Stage 4: Train
(CivitAI LoRAs)      (Flux + LoRA)      (ComfyUI I2V)      (musubi-tuner)
```

## Prerequisites

- **Ollama** (for prompt expansion) - https://ollama.com
- **ComfyUI** (for video generation) - see [ComfyUI Setup](#comfyui-setup) below
- Flux model (will download from HuggingFace by default)
- Wan2.1 models in `~/.daydream-scope/models/`:
  - `Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors` (DiT)
  - `Wan2.1-T2V-1.3B/Wan2.1_VAE.pth` (VAE)
  - `WanVideo_comfy/umt5-xxl-enc-fp8_e4m3fn.safetensors` (T5 text encoder)

## Project Structure

```
synthetic-wan-data/
├── src/
│   ├── stage1_scrape/          # CivitAI LoRA scraper + prompt expansion
│   ├── stage2_generate_images/ # Flux image dataset generation
│   └── stage3_generate_video/  # ComfyUI video dataset generation
├── musubi-tuner/               # Training (git submodule)
├── data/                       # Generated data (gitignored)
│   └── model_<id>/             # Per-model data
│       ├── images/             # Generated images + captions
│       ├── videos/             # Generated videos + captions
│       ├── cache/              # Cached latents
│       └── dataset_config.toml # Training config
├── train_command.txt           # Collection of training commands
└── .env                        # API keys (CIVITAI_API)
```

---

## Successful Example: model_245889 (ral-dissolve)

This configuration produced good results:

- **30 images** at 768x768, `num_repeats = 2`
- **20 videos** at 640x640, 33 frames, `num_repeats = 1`
- **64 epochs** total training (two runs of 32 epochs)
- Trigger word: `ddscope`

### Dataset Config (data/model_245889/dataset_config.toml)

```toml
[general]
caption_extension = ".txt"
batch_size = 1
enable_bucket = true
bucket_no_upscale = true

# Image dataset (30 images at 768x768)
[[datasets]]
image_directory = "C:/_dev/projects/synthetic-wan-data/data/model_245889/images"
cache_directory = "C:/_dev/projects/synthetic-wan-data/data/model_245889/cache/images"
resolution = [768, 768]
num_repeats = 2

# Video dataset (20 videos at 640x640, 33 frames)
[[datasets]]
video_directory = "C:/_dev/projects/synthetic-wan-data/data/model_245889/videos"
cache_directory = "C:/_dev/projects/synthetic-wan-data/data/model_245889/cache/videos"
resolution = [640, 640]
target_frames = [33]
frame_extraction = "head"
num_repeats = 1
```

---

## Critical Pitfalls

### 1. Video Frame Count Default (IMPORTANT!)

**If you don't specify `target_frames` in your dataset config, it defaults to `[1]` - meaning only 1 frame is extracted from each video!** This effectively turns video training into image training.

Always explicitly set `target_frames`:

```toml
[[datasets]]
video_directory = "..."
target_frames = [33]      # REQUIRED! Without this, only 1 frame is used
frame_extraction = "head" # Takes first N frames
```

Frame counts must be `N*4+1` format (1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, ..., 81).

**Frame extraction modes:**
- `head` (default): Takes first N frames from video
- `chunk`: Splits video into non-overlapping segments
- `slide`: Sliding window with stride
- `uniform`: Evenly spaced samples throughout video
- `full`: Uses all frames up to `max_frames` (ignores `target_frames`)

### 2. Continuing Training (--network_weights vs --resume)

**`--network_weights`** loads LoRA weights but **resets epoch counter to 0**. Checkpoints will overwrite previous ones!

**`--resume`** loads full training state (weights + optimizer + epoch counter), but requires `--save_state` during original training.

**To properly continue training:**

1. Add `--save_state` to your training command to save full state
2. Use `--resume ./output/model/state-XXXXX` to continue

**If you forgot `--save_state`:**
- Use different `--output_name` (e.g., `dissolve_v2`) to avoid overwriting
- Or rename existing checkpoints before continuing

### 3. Resolution and Memory

- 81 frames at 640x640 is very heavy - use 33 frames for faster training
- Images at 1024x1024 work but 768x768 is safer for memory
- Videos + images together work well when images have higher `num_repeats`

### 4. Caching Requirements

You must re-cache latents when you change:
- `target_frames`
- `resolution`
- Add/remove files from dataset

Clear cache and re-run:
```bash
rm -rf ../data/model_xxx/cache/*
uv run python src/musubi_tuner/wan_cache_latents.py --dataset_config "..."
uv run python src/musubi_tuner/wan_cache_text_encoder_outputs.py --dataset_config "..."
```

---

## Setup

### 1. Setup this environment (scraping + dataset generation)

```powershell
# Lightweight install (Stage 1 scraping only)
uv sync

# Full install (includes Stage 2+3 image/video generation - needs GPU)
uv sync --extra generate
```

### 2. Setup musubi-tuner environment (training)

```powershell
cd musubi-tuner
uv sync --extra cu130
```

### 3. ComfyUI Setup

Stage 3 uses ComfyUI for video generation. You need your own ComfyUI installation with:
- Wan2.1 or Wan2.2 models installed
- A working I2V (image-to-video) workflow

The included workflow (`src/stage3_generate_video/video_wan2_2_14B_i2v.json`) is based on the Wan2.2 I2V template. You may need to adjust node IDs in `generate_comfyui.py` if using a different workflow.

---

## Quick Start: run_pipeline.py

The unified `run_pipeline.py` handles all stages:

```bash
# Single model, all stages (scrape + images + videos)
uv run python run_pipeline.py --model-id 245889 --stages 1,2,3

# Multiple models, just images (no videos)
uv run python run_pipeline.py --model-id 245889 179353 180780 --stages 1,2

# Just videos (stages 1-2 already done)
uv run python run_pipeline.py --model-id 245889 --stages 3

# From file (one model ID per line)
uv run python run_pipeline.py --models-file models.txt --stages 1,2,3

# Custom counts
uv run python run_pipeline.py --model-id 245889 --stages 1,2,3 \
    --num-images 30 --num-videos 20

# Create dataset config (with correct target_frames!)
uv run python run_pipeline.py --model-id 245889 --create-config

# Output training commands
uv run python run_pipeline.py --model-id 245889 --training-commands
```

**Options:**
- `--stages`: Comma-separated stages to run (default: `1,2,3`)
- `--num-images`: Images to generate (default: 30)
- `--num-videos`: Videos to generate (default: 20)
- `--num-prompts`: Prompts to expand to (default: 50)
- `--target-frames`: Frames for video training config (default: 33)
- `--create-config`: Generate dataset_config.toml instead of running stages
- `--training-commands`: Output caching + training commands

---

## Running Stages Individually

### Stage 1: Scrape LoRA + Expand Prompts

Downloads the Flux LoRA, extracts prompt metadata, and expands to target count using Ollama.

```powershell
# Full pipeline: scrape + expand to 50 prompts (default)
uv run python -m src.stage1_scrape.scraper "https://civitai.com/models/317208/zombie-style-fluxsdxl"

# Custom expansion target
uv run python -m src.stage1_scrape.scraper "https://civitai.com/models/317208/zombie-style-fluxsdxl" \
    --expand-to 100
```

**Options:**
- `--expand-to`: Target number of prompts (default: 50, set to 0 to skip)
- `--ollama-model`: Ollama model for expansion (default: llama3.2)
- `--no-download`: Skip downloading the LoRA file

**Outputs:**
- `data/model_<id>/<filename>.safetensors` - The Flux LoRA file
- `data/model_<id>/prompts.json` - Prompts (scraped + generated)

### Stage 2: Generate Image Dataset

Uses Flux + a LoRA to generate training images with captions.

```powershell
uv run python -m src.stage2_generate_images.generate \
    --lora "./data/model_245889/ral-dissolve-flux.safetensors" \
    --prompts "./data/model_245889/prompts.json" \
    --output "./data/model_245889/images" \
    --num-images 30 \
    --width 1024 --height 1024
```

**Options:**
- `--lora`: Path to LoRA safetensors (required)
- `--prompts`: Path to prompts file or JSON (required)
- `--num-images`: Number of images to generate
- `--width` / `--height`: Resolution (default: 1024x1024)
- `--style-prefix` / `--style-suffix`: Add to all prompts
- `--seed`: Random seed for reproducibility

### Stage 3: Generate Video Dataset (ComfyUI)

Converts images to videos using ComfyUI with Wan2.1 I2V workflow.

**Make sure ComfyUI is running first!**

```powershell
uv run python -m src.stage3_generate_video.generate_comfyui \
    --input "./data/model_245889/images" \
    --output "./data/model_245889/videos" \
    --num-videos 20
```

The script automatically:
- Detects image aspect ratios and calculates appropriate video dimensions
- Uploads images to ComfyUI
- Injects prompts, seeds, and dimensions into the workflow
- Downloads completed videos
- Copies captions with trigger word prefix

**Options:**
- `--workflow`: Custom workflow JSON (default: `video_wan2_2_14B_i2v.json`)
- `--comfyui-url`: ComfyUI API URL (default: `http://127.0.0.1:8188`)
- `--num-videos`: Number of videos to generate
- `--max-pixels`: Max pixels for video dimensions (default: 409600 = 640x640)
- `--seed`: Random seed for reproducibility
- `--timeout`: Timeout per video in seconds (default: 600)

---

## Training (Stage 4)

### Create Dataset Config

Use `run_pipeline.py --create-config` or manually create `data/model_xxx/dataset_config.toml`:

```toml
[general]
caption_extension = ".txt"
batch_size = 1
enable_bucket = true
bucket_no_upscale = true

# Images - higher num_repeats for more style influence
[[datasets]]
image_directory = "C:/_dev/projects/synthetic-wan-data/data/model_xxx/images"
cache_directory = "C:/_dev/projects/synthetic-wan-data/data/model_xxx/cache/images"
resolution = [768, 768]
num_repeats = 2

# Videos - MUST specify target_frames!
[[datasets]]
video_directory = "C:/_dev/projects/synthetic-wan-data/data/model_xxx/videos"
cache_directory = "C:/_dev/projects/synthetic-wan-data/data/model_xxx/cache/videos"
resolution = [640, 640]
target_frames = [33]
frame_extraction = "head"
num_repeats = 1
```

### Cache Latents and Text Encoder Outputs

```powershell
cd musubi-tuner

uv run python src/musubi_tuner/wan_cache_latents.py \
    --dataset_config "../data/model_xxx/dataset_config.toml" \
    --vae "C:/Users/ryanf/.daydream-scope/models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"

uv run python src/musubi_tuner/wan_cache_text_encoder_outputs.py \
    --dataset_config "../data/model_xxx/dataset_config.toml" \
    --t5 "C:/Users/ryanf/.daydream-scope/models/WanVideo_comfy/umt5-xxl-enc-fp8_e4m3fn.safetensors"
```

### Train

```powershell
uv run accelerate launch --mixed_precision bf16 src/musubi_tuner/wan_train_network.py \
    --task t2v-1.3B \
    --dit "C:/Users/ryanf/.daydream-scope/models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors" \
    --dataset_config "../data/model_xxx/dataset_config.toml" \
    --output_dir "./output/model_xxx" \
    --output_name stylename \
    --sdpa \
    --network_module networks.lora_wan \
    --network_dim 64 \
    --optimizer_type adamw8bit \
    --learning_rate 2e-4 \
    --gradient_checkpointing \
    --timestep_sampling shift \
    --discrete_flow_shift 3.0 \
    --max_train_epochs 32 \
    --save_every_n_epochs 8 \
    --save_state \
    --seed 42
```

**Key options:**
- `--network_dim`: LoRA rank (64 recommended)
- `--learning_rate`: `2e-4` for image+video training
- `--max_train_epochs`: 32-64 epochs for good results
- `--save_state`: **Add this to enable proper training continuation!**
- `--save_every_n_epochs`: Checkpoint frequency

### Continue Training (if needed)

**With `--save_state` (proper resume):**
```powershell
uv run accelerate launch ... \
    --resume "./output/model_xxx/state-XXXXX" \
    --max_train_epochs 64
```

**Without `--save_state` (weights only, epoch resets to 0):**
```powershell
uv run accelerate launch ... \
    --network_weights "./output/model_xxx/stylename-000032.safetensors" \
    --output_name stylename_v2 \
    --max_train_epochs 32
```

---

## Quick Reference: Training Commands

See `train_command.txt` for a collection of caching and training commands for all models.

---

## Environment Variables

Create a `.env` file:

```
CIVITAI_API=your_api_key_here
```

Get your API key from https://civitai.com/user/account
