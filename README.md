# Distill Flux LoRAs to Wan2.1 LoRAs

Four-stage pipeline for distilling Flux LoRA styles into Wan2.1 video LoRAs.

```
Stage 1: Scrape   -> Stage 2: Images -> Stage 3: Videos -> Stage 4: Train
(CivitAI LoRAs)      (Flux + LoRA)      (LTX-2 I2V)        (musubi-tuner)
```

## Prerequisites

- **Ollama** (for prompt expansion) - https://ollama.com
- Flux model (will download from HuggingFace by default)
- LTX-2 model (will download from HuggingFace by default)
- Wan2.1 models in `~/.daydream-scope/models/`:
  - `Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors` (DiT)
  - `Wan2.1-T2V-1.3B/Wan2.1_VAE.pth` (VAE)
  - `Wan2.1-T2V-1.3B/google/umt5-xxl/` (tokenizer)
  - `WanVideo_comfy/umt5-xxl-enc-fp8_e4m3fn.safetensors` (T5 text encoder)

## Project Structure

```
synthetic-wan-data/
├── src/
│   ├── stage1_scrape/          # CivitAI LoRA scraper + prompt expansion
│   ├── stage2_generate_images/ # Flux image dataset generation
│   └── stage3_generate_video/  # LTX-2 video dataset generation
├── musubi-tuner/               # Training (git submodule)
├── data/                       # Generated data (gitignored)
│   ├── loras/                  # Downloaded LoRA files
│   ├── prompts/                # Scraped prompt metadata
│   ├── datasets/               # Image datasets
│   └── video_datasets/         # Video datasets
├── examples/                   # Example prompts
└── .env                        # API keys (CIVITAI_API)
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

---

## Stage 1: Scrape LoRA + Expand Prompts

Downloads the Flux LoRA, extracts prompt metadata, and expands to target count using Ollama.

```powershell
# Full pipeline: scrape + expand to 50 prompts (default)
uv run python -m src.stage1_scrape.scraper "https://civitai.com/models/317208/zombie-style-fluxsdxl"

# Custom expansion target
uv run python -m src.stage1_scrape.scraper "https://civitai.com/models/317208/zombie-style-fluxsdxl" \
    --expand-to 100

# Skip expansion (for testing)
uv run python -m src.stage1_scrape.scraper "https://civitai.com/models/317208/zombie-style-fluxsdxl" \
    --expand-to 0

# Dry run (no download, no expansion)
uv run python -m src.stage1_scrape.scraper "https://civitai.com/models/317208/zombie-style-fluxsdxl" \
    --no-download --expand-to 0
```

**Options:**
- `--expand-to`: Target number of prompts (default: 50, set to 0 to skip)
- `--ollama-model`: Ollama model for expansion (default: llama3.2)
- `--no-download`: Skip downloading the LoRA file

**Outputs:**
- `data/loras/<filename>.safetensors` - The Flux LoRA file
- `data/prompts/<model_id>_prompts.json` - Prompts (scraped + generated)

**Note:** Requires Ollama running locally (`ollama serve`).

---

## Stage 2: Generate Image Dataset

Uses Flux + a LoRA to generate training images with captions.

```powershell
uv run python -m src.stage2_generate_images.generate \
    --lora "./data/loras/zombieStyleFlux_v1.safetensors" \
    --prompts "./data/prompts/317208_prompts.json" \
    --output "./data/datasets/zombie"
```

Or with a custom prompts file:

```powershell
uv run python -m src.stage2_generate_images.generate \
    --lora "./data/loras/my_lora.safetensors" \
    --prompts "./examples/ral_dissolve_prompts.txt" \
    --output "./data/datasets/dissolve" \
    --seed 42
```

**Output:**
```
data/datasets/dissolve/
├── 0000.png, 0001.png, ...   # Images
├── 0000.txt, 0001.txt, ...   # Captions
├── cache/                     # Will contain cached latents
├── metadata.json              # Generation parameters
└── dataset_config.toml        # Ready for musubi-tuner
```

**Options:**
- `--flux`: Flux model (default: `black-forest-labs/FLUX.1-dev`)
- `--lora`: Path to LoRA safetensors (required)
- `--prompts`: Path to prompts file or JSON (required)
- `--num_images`: Number of images (default: number of prompts)
- `--lora_scale`: LoRA strength (default: 1.0)
- `--style_prefix` / `--style_suffix`: Add to all prompts
- `--width` / `--height`: Resolution (default: 1024x1024)
- `--seed`: Random seed for reproducibility

---

## Stage 3: Generate Video Dataset

Converts the image dataset to videos using LTX-2 Image-to-Video.

```powershell
uv run python -m src.stage3_generate_video.generate \
    --input "./data/datasets/zombie" \
    --output "./data/video_datasets/zombie"
```

**Output:**
```
data/video_datasets/zombie/
├── 0000.mp4, 0001.mp4, ...   # Videos
├── 0000.txt, 0001.txt, ...   # Captions
├── cache/                     # Will contain cached latents
├── metadata.json              # Generation parameters
└── dataset_config.toml        # Ready for musubi-tuner
```

**Options:**
- `--model`: LTX-2 model (default: `Lightricks/LTX-2`)
- `--num-frames`: Frames per video, must be 8*N+1 (default: 121 = 5 seconds @ 24fps)
- `--frame-rate`: Frame rate (default: 24.0)
- `--width` / `--height`: Video resolution (default: 768x512)
- `--steps`: Inference steps (default: 40)
- `--guidance-scale`: CFG scale (default: 4.0)
- `--seed`: Random seed for reproducibility
- `--max-videos`: Limit number of videos (for testing)

---

## Stage 4: Train LoRA

Train a Wan2.1 LoRA on the video dataset using musubi-tuner.

### 4a. Cache Latents

Pre-encode videos to VAE latents (done once so training doesn't re-encode every epoch):

```powershell
cd musubi-tuner

uv run python src/musubi_tuner/wan_cache_latents.py \
    --dataset_config "../data/video_datasets/zombie/dataset_config.toml" \
    --vae "C:/Users/ryanf/.daydream-scope/models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"
```

### 4b. Cache Text Encoder Outputs

Pre-encode captions through T5 (also done once):

```powershell
uv run python src/musubi_tuner/wan_cache_text_encoder_outputs.py \
    --dataset_config "../data/video_datasets/zombie/dataset_config.toml" \
    --t5 "C:/Users/ryanf/.daydream-scope/models/WanVideo_comfy/umt5-xxl-enc-fp8_e4m3fn.safetensors"
```

### 4c. Train

```powershell
uv run accelerate launch --mixed_precision bf16 src/musubi_tuner/wan_train_network.py \
    --task t2v-1.3B \
    --dit "C:/Users/ryanf/.daydream-scope/models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors" \
    --vae "C:/Users/ryanf/.daydream-scope/models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth" \
    --t5 "C:/Users/ryanf/.daydream-scope/models/WanVideo_comfy/umt5-xxl-enc-fp8_e4m3fn.safetensors" \
    --dataset_config "../data/video_datasets/zombie/dataset_config.toml" \
    --output_dir "./output/zombie_lora" \
    --output_name zombie \
    --sdpa \
    --network_module networks.lora_wan \
    --network_dim 64 \
    --optimizer_type adamw8bit \
    --learning_rate 2e-4 \
    --gradient_checkpointing \
    --timestep_sampling shift \
    --discrete_flow_shift 3.0 \
    --max_train_epochs 16 \
    --save_every_n_epochs 4 \
    --seed 42
```

**Key options:**
- `--sdpa`: Use PyTorch scaled dot product attention
- `--network_dim`: LoRA rank (64-128 recommended)
- `--learning_rate`: Use `2e-4` for video training
- `--max_train_epochs`: Number of epochs (16+ for video)
- `--save_every_n_epochs`: Checkpoint frequency

---

## Environment Variables

Create a `.env` file:

```
CIVITAI_API=your_api_key_here
```

Get your API key from https://civitai.com/user/account
