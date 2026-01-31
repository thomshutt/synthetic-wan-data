# Distill Flux LoRAs → Wan2.1 AR-Compatible LoRAs

Generate style datasets from Flux LoRAs, then train Wan2.1 LoRAs with causal attention for autoregressive model compatibility (LongLive, etc.).

## Prerequisites

- Flux model (will download from HuggingFace by default)
- Wan2.1 models in `~/.daydream-scope/models/`:
  - `Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors` (DiT)
  - `Wan2.1-T2V-1.3B/Wan2.1_VAE.pth` (VAE)
  - `Wan2.1-T2V-1.3B/google/umt5-xxl/` (tokenizer)
  - `WanVideo_comfy/umt5-xxl-enc-fp8_e4m3fn.safetensors` (T5 text encoder)

---

## Setup

### 1. Setup this environment (dataset generation)

```powershell
cd C:\_dev\projects\scope\scripts\distill_flux_lora
uv sync
```

### 2. Setup musubi-tuner environment (training)

```powershell
cd C:\_dev\projects\scope\notes\musubi-tuner
uv sync --extra cu130
```

---

## Step 1: Generate Image Dataset

Use Flux + a style LoRA to generate training images with captions.

```powershell
cd C:\_dev\projects\scope\scripts\distill_flux_lora

uv run python generate_dataset.py --lora "C:\_dev\models\comfyui_models\loras\FLUX\ral-dissolve-flux.safetensors" --prompts "./example_prompts.txt" --output "./datasets/dissolve" --seed 42
```

**Output:**
```
datasets/dissolve/
├── 0000.png, 0001.png, ...   # Images (viewable)
├── 0000.txt, 0001.txt, ...   # Captions
├── cache/                     # Will contain cached latents
├── metadata.json              # Generation parameters
└── dataset_config.toml        # Ready for musubi-tuner
```

**Options:**
- `--flux`: Flux model (default: `black-forest-labs/FLUX.1-dev` from HuggingFace)
- `--lora`: Path to Flux LoRA safetensors (required)
- `--prompts`: Path to prompts file, one per line (required)
- `--output`: Output directory (default: `./dataset`)
- `--num_images`: Number of images (default: number of prompts)
- `--lora_scale`: LoRA strength (default: 1.0)
- `--style_prefix` / `--style_suffix`: Add to all prompts
- `--width` / `--height`: Resolution (default: 1024x1024)
- `--seed`: Random seed for reproducibility

---

## Step 2 (Alternative): Train on Images Only

If you want to skip video generation and train directly on images:

**Cache latents:**
```powershell
cd C:\_dev\projects\scope\notes\musubi-tuner

uv run python src/musubi_tuner/wan_cache_latents.py --dataset_config "C:/_dev/projects/scope/scripts/distill_flux_lora/datasets/dissolve/dataset_config.toml" --vae "C:/Users/ryanf/.daydream-scope/models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"
```

**Cache text encoder:**
```powershell
uv run python src/musubi_tuner/wan_cache_text_encoder_outputs.py --dataset_config "C:/_dev/projects/scope/scripts/distill_flux_lora/datasets/dissolve/dataset_config.toml" --t5 "C:/Users/ryanf/.daydream-scope/models/WanVideo_comfy/umt5-xxl-enc-fp8_e4m3fn.safetensors"
```

**Train (longer run, recommended for small datasets):**
```powershell
uv run accelerate launch --mixed_precision bf16 src/musubi_tuner/wan_train_network.py --task t2v-1.3B --dit "C:/Users/ryanf/.daydream-scope/models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors" --vae "C:/Users/ryanf/.daydream-scope/models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth" --t5 "C:/Users/ryanf/.daydream-scope/models/WanVideo_comfy/umt5-xxl-enc-fp8_e4m3fn.safetensors" --dataset_config "C:/_dev/projects/scope/scripts/distill_flux_lora/datasets/dissolve/dataset_config.toml" --output_dir "./output/dissolve_lora_long" --output_name dissolve-long --sdpa --network_module networks.lora_wan --network_dim 64 --optimizer_type adamw8bit --learning_rate 1e-4 --gradient_checkpointing --timestep_sampling shift --discrete_flow_shift 3.0 --max_train_epochs 64 --save_every_n_epochs 8 --seed 42
```

**Tips for image-only training:**
- Lower learning rate (`1e-4`) for longer runs to avoid overfitting
- More epochs (`64+`) since images train faster than video
- Increase `num_repeats` in `dataset_config.toml` to effectively expand small datasets
- Save checkpoints frequently to find the sweet spot before overfitting

---

## Step 2: Generate Video Dataset

Convert images to videos using Wan I2V. This creates video training data where causal attention is meaningful.

```powershell
cd C:\_dev\projects\scope\scripts\distill_flux_lora

uv run python generate_video_dataset.py --input "./datasets/dissolve" --output "./datasets/dissolve_video" --model "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers" --seed 42
```

**Options:**
- `--model`: I2V model to use (options below)
- `--num_frames`: Frames per video (default: 81, must be N*4+1)
- `--height` / `--width`: Video resolution (default: 480x832)
- `--max_videos`: Limit number of videos to generate (for testing)

**Available models (diffusers format):**
- `Wan-AI/Wan2.2-I2V-A14B-Diffusers` - Latest MoE architecture, best quality
- `Wan-AI/Wan2.1-I2V-14B-720P-Diffusers` - 720P resolution
- `Wan-AI/Wan2.1-I2V-14B-480P-Diffusers` - 480P resolution (default, faster)

---

## Step 3: Cache Latents

Pre-encode videos to VAE latents. Done once so training doesn't re-encode every epoch.

```powershell
cd C:\_dev\projects\scope\notes\musubi-tuner

uv run python src/musubi_tuner/wan_cache_latents.py --dataset_config "C:/_dev/projects/scope/scripts/distill_flux_lora/datasets/dissolve_video/dataset_config.toml" --vae "C:/Users/ryanf/.daydream-scope/models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"
```

---

## Step 4: Cache Text Encoder Outputs

Pre-encode captions through T5. Also done once.

```powershell
cd C:\_dev\projects\scope\notes\musubi-tuner

uv run python src/musubi_tuner/wan_cache_text_encoder_outputs.py --dataset_config "C:/_dev/projects/scope/scripts/distill_flux_lora/datasets/dissolve_video/dataset_config.toml" --t5 "C:/Users/ryanf/.daydream-scope/models/WanVideo_comfy/umt5-xxl-enc-fp8_e4m3fn.safetensors"
```

---

## Step 5: Train LoRA

Train a Wan2.1 LoRA on the video dataset.

**With causal attention (for AR models like LongLive):**
```powershell
cd C:\_dev\projects\scope\notes\musubi-tuner

uv run accelerate launch --mixed_precision bf16 src/musubi_tuner/wan_train_network.py --task t2v-1.3B --dit "C:/Users/ryanf/.daydream-scope/models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors" --vae "C:/Users/ryanf/.daydream-scope/models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth" --t5 "C:/Users/ryanf/.daydream-scope/models/WanVideo_comfy/umt5-xxl-enc-fp8_e4m3fn.safetensors" --dataset_config "C:/_dev/projects/scope/scripts/distill_flux_lora/datasets/dissolve_video/dataset_config.toml" --output_dir "./output/dissolve_lora" --output_name dissolve-causal --causal_attention --sdpa --network_module networks.lora_wan --network_dim 64 --optimizer_type adamw8bit --learning_rate 2e-4 --gradient_checkpointing --timestep_sampling shift --discrete_flow_shift 3.0 --max_train_epochs 16 --save_every_n_epochs 4 --seed 42
```

**Without causal attention (standard bidirectional):**
```powershell
cd C:\_dev\projects\scope\notes\musubi-tuner

uv run accelerate launch --mixed_precision bf16 src/musubi_tuner/wan_train_network.py --task t2v-1.3B --dit "C:/Users/ryanf/.daydream-scope/models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors" --vae "C:/Users/ryanf/.daydream-scope/models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth" --t5 "C:/Users/ryanf/.daydream-scope/models/WanVideo_comfy/umt5-xxl-enc-fp8_e4m3fn.safetensors" --dataset_config "C:/_dev/projects/scope/scripts/distill_flux_lora/datasets/dissolve_video/dataset_config.toml" --output_dir "./output/dissolve_lora" --output_name dissolve-standard --sdpa --network_module networks.lora_wan --network_dim 64 --optimizer_type adamw8bit --learning_rate 2e-4 --gradient_checkpointing --timestep_sampling shift --discrete_flow_shift 3.0 --max_train_epochs 16 --save_every_n_epochs 4 --seed 42
```

**Key options:**
- `--causal_attention`: Enables causal masking (experimental, for AR compatibility)
- `--sdpa`: Use PyTorch scaled dot product attention
- `--network_dim`: LoRA rank (64-128 recommended)
- `--max_train_epochs`: Number of epochs (use 1 for quick test, 16+ for real training)
- `--save_every_n_epochs`: Checkpoint frequency

---

## Notes

### Why Video Training?

Training on videos (vs images) allows the LoRA to learn temporal patterns. With causal attention, the model learns to generate style using only past frames - matching how autoregressive models like LongLive actually generate.

### Why Higher LoRA Rank?

AR models have less context (only past frames). Higher rank LoRAs (64-128 vs typical 32) give more capacity to capture style effectively.

---

## Sources

- [Best Wan Models 2026](https://www.siliconflow.com/articles/en/the-best-wan-models-in-2025)
- [Open Source Video Generation Models](https://www.hyperstack.cloud/blog/case-study/best-open-source-video-generation-models)
