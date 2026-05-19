# FLUX.1-dev Training Guide

This guide covers LoRA fine-tuning of [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) using VeOmni, including dataset preparation, multi-GPU training with Ulysses Sequence Parallelism (SP), and inference with trained adapters.

---

## 1. Environment Setup

```shell
uv sync --extra gpu --dev
source .venv/bin/activate
```

---

## 2. Download Model

```shell
python3 scripts/download_hf_model.py \
    --repo_id black-forest-labs/FLUX.1-dev \
    --local_dir ./FLUX.1-dev
```

The downloaded directory has the following structure:

```
FLUX.1-dev/
├── ae.safetensors                    # VAE autoencoder
├── flux1-dev.safetensors             # DiT transformer weights
├── scheduler/
│   └── scheduler_config.json
├── text_encoder/                     # CLIP text encoder
│   ├── config.json
│   └── model.safetensors
├── text_encoder_2/                   # T5 text encoder
│   ├── config.json
│   └── model.safetensors
├── tokenizer/                        # CLIP tokenizer
│   ├── merges.txt
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   └── vocab.json
├── tokenizer_2/                      # T5 tokenizer
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   └── spiece.model
└── transformer/                      # DiT transformer config
    └── config.json
```

---

## 3. Prepare Dataset

VeOmni supports two training workflows:

| Workflow | `training_task` | Description |
|---|---|---|
| **Offline** (recommended) | `offline_training` | Pre-embed images once; re-use embeddings across epochs. Saves GPU memory during training. |
| **Online** | `online_training` | Embed images on-the-fly each step. Requires the VAE + text encoders to stay on GPU throughout training. |

### 3.1 Prepare your image-text dataset

Organize your dataset as a directory with `captions.txt` and `images.txt`:

```
your-dataset/
├── captions.txt   # one caption per line
├── images.txt     # one relative image path per line (mirrors captions.txt)
└── images/        # image files
```

Each line in `captions.txt` corresponds to the same line in `images.txt`:

```
# captions.txt
A watercolor painting of a sunset over mountains
A portrait of a cat sitting on a windowsill

# images.txt
images/sunset_001.jpg
images/cat_002.png
```

### 3.2 Convert to VeOmni Parquet format

The conversion script reads `captions.txt` and `images.txt`, loads each image as raw bytes, and writes sharded Parquet files (`0.parquet`, `1.parquet`, …) with columns `text`, `image_bytes`, and `source`.

```shell
python3 scripts/multimodal/convert_data/flux_image_dataset.py \
    --dataset_path ./your-dataset \
    --output_dir   ./your-dataset-parquet
```

### 3.3 Offline Workflow (recommended)

#### Step 1 – Run offline embedding (once)

This step encodes every image with the VAE and every caption with the CLIP + T5 text encoders, saving the embeddings as Parquet shards. It only needs to run once per dataset.

```shell
NPROC_PER_NODE=4 bash train.sh tasks/train_dit.py configs/dit/flux_lora_dit.yaml \
    --model.model_path           ./FLUX.1-dev/flux1-dev.safetensors \
    --model.condition_model_path ./FLUX.1-dev \
    --data.train_path            ./your-dataset-parquet \
    --data.source_name           X2I-text-to-image \
    --data.offline_embedding_save_dir ./your-dataset_offline \
    --train.training_task        offline_embedding \
    --train.global_batch_size    4 \
    --train.accelerator.ulysses_size 1
```

The resulting `your-dataset_offline/` directory contains `rank_N_shard_M.parquet` files. Each row stores three pickled tensors:

| Column | Shape | Description |
|---|---|---|
| `latents` | `(1, 32, H, W)` | VAE posterior parameters (mean + log-variance concatenated along the channel axis; `32 = 2 × 16`) |
| `context` | `(1, 512, 4096)` | T5 text embedding |
| `pooled_context` | `(1, 768)` | CLIP pooled text embedding |

#### Step 2 – Train on the offline dataset

```shell
SP_SIZE=2
NPROC_PER_NODE=8   # 4 DP replicas × SP_SIZE=2

bash train.sh tasks/train_dit.py configs/dit/flux_lora_dit.yaml \
    --model.model_path           ./FLUX.1-dev/flux1-dev.safetensors \
    --model.condition_model_path ./FLUX.1-dev \
    --data.train_path            ./your-dataset_offline \
    --data.source_name           X2I-text-to-image \
    --train.training_task        offline_training \
    --train.global_batch_size    8 \
    --train.micro_batch_size     1 \
    --train.accelerator.ulysses_size ${SP_SIZE} \
    --train.checkpoint.output_dir ./exp/FLUX.1-dev_lora \
    --train.checkpoint.save_hf_weights true \
    --train.checkpoint.save_epochs 5 \
    --train.checkpoint.load_path auto \
    --train.num_train_epochs 30 \
    --train.wandb.enable false
```

### 3.4 Online Workflow

Pass raw Parquet images directly during training. The VAE and text encoders run each step.

```shell
NPROC_PER_NODE=4 bash train.sh tasks/train_dit.py configs/dit/flux_lora_dit.yaml \
    --model.model_path           ./FLUX.1-dev/flux1-dev.safetensors \
    --model.condition_model_path ./FLUX.1-dev \
    --data.train_path            ./your-dataset-parquet \
    --data.source_name           X2I-text-to-image \
    --data.mm_configs.height     1024 \
    --data.mm_configs.width      768 \
    --train.training_task        online_training \
    --train.global_batch_size    4 \
    --train.micro_batch_size     1 \
    --train.accelerator.ulysses_size 1 \
    --train.checkpoint.output_dir ./exp/FLUX.1-dev_lora \
    --train.checkpoint.save_hf_weights true \
    --train.num_train_epochs 30
```

---

## 4. Training Configuration

The default LoRA config (`configs/dit/flux_lora_dit.yaml`) targets the attention and feed-forward projections in both joint and single transformer blocks:

```yaml
model:
  lora_config:
    rank: 16
    alpha: 16
    lora_modules:
      - a_to_qkv
      - b_to_qkv
      - ff_a.0
      - ff_a.2
      - ff_b.0
      - ff_b.2
      - a_to_out
      - b_to_out
      - proj_out
      - norm.linear
      - norm1_a.linear
      - norm1_b.linear
      - to_qkv_mlp
```

### Sequence Parallelism (SP)

VeOmni supports Ulysses SP for long image sequences. SP splits the sequence dimension across GPUs within each data-parallel replica, reducing per-GPU memory while keeping training numerically equivalent to SP=1.

| `ulysses_size` | GPUs (with 4 DP replicas) |
|---|---|
| 1 | 4 |
| 2 | 8 |

Set `--train.accelerator.ulysses_size` to enable SP. The loss and gradient norms are aligned between SP=1 and SP=2 at equal DP sizes.

---

## 5. Checkpoint Output

When `--train.checkpoint.save_hf_weights true` is set, each save produces a directory compatible with `load_lora_adapter`:

```
exp/FLUX.1-dev_lora/checkpoints/
└── global_step_200/
    ├── adapter_config.json
    └── adapter_model.safetensors
```

---

## 6. Inference

### 6.1 Base model (no LoRA)

```python
import torch
from diffusers import FluxPipeline

model_id = "./FLUX.1-dev"

pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
pipe.to("cuda")

prompt = "A watercolor painting of a sunset over mountains, soft brush strokes, warm colors"

image = pipe(
    prompt=prompt,
    height=1024,
    width=768,
    guidance_scale=3.5,
    num_inference_steps=50,
).images[0]

image.save("output.png")
```

### 6.2 With trained LoRA adapter

```python
import torch
from diffusers import FluxPipeline

model_id = "./FLUX.1-dev"
lora_dir = "./exp/FLUX.1-dev_lora/checkpoints/global_step_200"

pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
pipe.to("cuda")

pipe.load_lora_adapter(lora_dir, prefix="base_model.model", adapter_name="flux_lora")
pipe.set_adapters("flux_lora", adapter_weights=1.0)  # adjust strength between 0.5–1.0

prompt = "A watercolor painting of a sunset over mountains, soft brush strokes, warm colors"

image = pipe(
    prompt=prompt,
    height=1024,
    width=768,
    guidance_scale=3.5,
    num_inference_steps=50,
).images[0]

image.save("output_lora.png")
```

---

## 7. Architecture Overview

FLUX.1-dev is a diffusion transformer (DiT) model that uses a **dual-stream** architecture with joint transformer blocks and single-stream blocks. VeOmni splits training into two independent models:

```
┌─────────────────────────────────────────────────────────────────┐
│  FluxConditionModel  (frozen, not parallelized)                  │
│  ─ encodes raw images into latents using the VAE                 │
│  ─ encodes text prompts using CLIP + T5 text encoders            │
│  ─ samples noise, timesteps, and builds the training targets     │
│  ─ loaded from the same FLUX.1-dev checkpoint directory          │
└─────────────────────┬───────────────────────────────────────────┘
                      │  get_condition() + process_condition()
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│  FluxModel  (trainable, FSDP + SP-parallelized)                  │
│  ─ the core DiT backbone (19 joint + 38 single blocks)           │
│  ─ registered as a transformers PreTrainedModel for FSDP         │
│  ─ forward() computes loss from conditioned inputs               │
└─────────────────────────────────────────────────────────────────┘
```

The two models are linked by `FluxConfig.condition_model_type = "FluxConditionModel"`, which tells `DiTTrainer` which condition model class to instantiate.

### Key differences from WAN2.1

| Aspect | WAN2.1 (Video) | FLUX.1 (Image) |
|---|---|---|
| Input | Video frames `(B, C, F, H, W)` | Image `(B, C, H, W)` |
| Text encoder | UMT5 (single) | CLIP + T5 (dual) |
| Pooled embedding | Not used | CLIP pooled embedding |
| Guidance | Not embedded | Guidance embedder (scale=3.5) |
| Transformer blocks | Single type | Joint (19) + Single (38) |
| Positional encoding | 3D RoPE (time + height + width) | 2D RoPE (height + width) |
| Flow matching shift | 5.0 | 1.0 |
