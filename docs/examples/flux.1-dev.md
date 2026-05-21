# FLUX.1-dev Training Guide

This guide covers SFT and LoRA fine-tuning of [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) using VeOmni, including dataset preparation, multi-GPU training with FSDP2, and inference with trained checkpoints.

---

## 1. Environment Setup

```shell
uv sync --extra gpu --dev
source .venv/bin/activate
```

---

## 2. Download Model

Download FLUX.1-dev from Hugging Face in diffusers format:

```shell
python3 scripts/download_hf_model.py \
    --repo_id black-forest-labs/FLUX.1-dev \
    --local_dir ./FLUX.1-dev
```

The downloaded directory has the following structure:

```
FLUX.1-dev/
├── scheduler/
│   └── scheduler_config.json
├── tokenizer/
│   ├── vocab.json, merges.txt, ...
├── tokenizer_2/
│   ├── spiece.model, ...
├── text_encoder/
│   ├── config.json, model.safetensors
├── text_encoder_2/
│   ├── config.json, model.safetensors
├── transformer/
│   ├── config.json, model.safetensors
├── vae/
│   ├── config.json, model.safetensors
└── model_index.json
```

---

## 3. Prepare Dataset

VeOmni supports two training workflows for Flux:

| Workflow | `training_task` | Description |
|---|---|---|
| **Offline** (recommended) | `offline_training` | Pre-embed images once; re-use embeddings across epochs. Saves GPU memory during training. |
| **Online** | `online_training` | Embed images on-the-fly each step. Requires the VAE + text encoders to stay on GPU throughout training. |

### 3.1 Prepare image-text data

Convert your image-text dataset to VeOmni Parquet format. Each row should contain:
- `text` or `prompt`: the caption
- `image_bytes` or `image_path`: the target image

Example conversion script:

```shell
python3 scripts/multimodal/convert_data/flux_image_dataset.py \
    --dataset_path ./your_image_dataset \
    --output_dir   ./flux_image_train
```

### 3.2 Offline Workflow (recommended)

#### Step 1 – Run offline embedding (once)

This step encodes every image with the VAE and every caption with the CLIP + T5 text encoders, saving the embeddings as Parquet shards. It only needs to run once per dataset.

```shell
NPROC_PER_NODE=4 bash train.sh tasks/train_dit.py configs/dit/flux_sft.yaml \
    --model.model_path           ./FLUX.1-dev/transformer \
    --model.condition_model_path ./FLUX.1-dev \
    --data.train_path            ./flux_image_train \
    --data.source_name           Flux \
    --data.offline_embedding_save_dir ./flux_image_train_offline \
    --train.training_task        offline_embedding \
    --train.global_batch_size    4 \
    --train.accelerator.ulysses_size 1
```

The resulting `flux_image_train_offline/` directory contains `rank_N_shard_M.parquet` files. Each row stores pickled tensors:

| Column | Shape | Description |
|---|---|---|
| `latents` | `(1, 32, H, W)` | VAE posterior parameters (mean + log-variance concatenated; `32 = 2 × 16`) |
| `context` | `(1, 512, 4096)` | T5 text embedding |
| `pooled_context` | `(1, 768)` | CLIP pooled text embedding |

#### Step 2 – Train on the offline dataset

```shell
NPROC_PER_NODE=4 bash train.sh tasks/train_dit.py configs/dit/flux_sft.yaml \
    --model.model_path           ./FLUX.1-dev/transformer \
    --model.condition_model_path ./FLUX.1-dev \
    --data.train_path            ./flux_image_train_offline \
    --data.source_name           Flux \
    --train.training_task        offline_training \
    --train.global_batch_size    8 \
    --train.micro_batch_size     1 \
    --train.accelerator.ulysses_size 1 \
    --train.checkpoint.output_dir ./exp/FLUX.1-dev_sft \
    --train.checkpoint.save_hf_weights true \
    --train.checkpoint.save_epochs 1 \
    --train.checkpoint.load_path auto \
    --train.num_train_epochs 3 \
    --train.wandb.enable false
```

### 3.3 Online Workflow

Pass raw Parquet images directly during training. The VAE and text encoders run each step.

```shell
NPROC_PER_NODE=4 bash train.sh tasks/train_dit.py configs/dit/flux_sft.yaml \
    --model.model_path           ./FLUX.1-dev/transformer \
    --model.condition_model_path ./FLUX.1-dev \
    --data.train_path            ./flux_image_train \
    --data.source_name           Flux \
    --train.training_task        online_training \
    --train.global_batch_size    4 \
    --train.micro_batch_size     1 \
    --train.accelerator.ulysses_size 1 \
    --train.checkpoint.output_dir ./exp/FLUX.1-dev_sft \
    --train.checkpoint.save_hf_weights true \
    --train.num_train_epochs 3
```

---

## 4. LoRA Fine-tuning

LoRA fine-tuning targets the attention and feed-forward projections of the Flux transformer:

```shell
NPROC_PER_NODE=4 bash train.sh tasks/train_dit.py configs/dit/flux_lora.yaml \
    --model.model_path           ./FLUX.1-dev/transformer \
    --model.condition_model_path ./FLUX.1-dev \
    --data.train_path            ./flux_image_train \
    --data.source_name           Flux \
    --train.training_task        online_training \
    --train.global_batch_size    4 \
    --train.accelerator.ulysses_size 1 \
    --train.checkpoint.output_dir ./exp/FLUX.1-dev_lora \
    --train.checkpoint.save_hf_weights true \
    --train.num_train_epochs 3
```

---

## 5. Training Configuration

### Key Config Fields

| Field | Default | Description |
|---|---|---|
| `model.condition_model_cfg.max_sequence_length` | 512 | Maximum T5 text encoding length |
| `model.condition_model_cfg.height` | 1024 | Training image height |
| `model.condition_model_cfg.width` | 768 | Training image width |
| `model.condition_model_cfg.guidance_scale` | 3.5 | CFG guidance scale for training |
| `model.condition_model_cfg.shift` | 1.15 | Flow matching shift parameter |

### LoRA Config

The default LoRA config targets attention and feed-forward layers:

```yaml
model:
  lora_target_modules: to_q,to_k,to_v,to_out.0,ff.net.0.proj,ff.net.2
  lora_alpha: 16.0
  lora_rank: 16
```

---

## 6. Checkpoint Output

When `--train.checkpoint.save_hf_weights true` is set, each save produces a directory compatible with diffusers:

```
exp/FLUX.1-dev_sft/checkpoints/
└── global_step_500/
    ├── config.json
    └── model.safetensors
```

The saved checkpoint can be loaded directly by diffusers:

```python
from diffusers import FluxTransformer2DModel

transformer = FluxTransformer2DModel.from_pretrained(
    "./exp/FLUX.1-dev_sft/checkpoints/global_step_500"
)
```

---

## 7. Inference

### 7.1 Base model (no fine-tuning)

```python
import torch
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")

prompt = "A cat sitting on a windowsill, watercolor style"
image = pipe(
    prompt=prompt,
    height=1024,
    width=768,
    guidance_scale=3.5,
    num_inference_steps=50,
).images[0]
image.save("output.png")
```

### 7.2 With fine-tuned transformer

```python
import torch
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")

# Swap in the fine-tuned transformer
from diffusers import FluxTransformer2DModel

pipe.transformer = FluxTransformer2DModel.from_pretrained(
    "./exp/FLUX.1-dev_sft/checkpoints/global_step_500",
    torch_dtype=torch.bfloat16,
)
pipe.transformer.to("cuda")

prompt = "A cat sitting on a windowsill, watercolor style"
image = pipe(
    prompt=prompt,
    height=1024,
    width=768,
    guidance_scale=3.5,
    num_inference_steps=50,
).images[0]
image.save("output_finetuned.png")
```

### 7.3 With LoRA adapter

```python
import torch
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")

pipe.transformer.load_lora_adapter(
    "./exp/FLUX.1-dev_lora/checkpoints/global_step_500",
    prefix="base_model.model",
    adapter_name="flux_lora",
)
pipe.set_adapters("flux_lora", adapter_weights=1.0)

prompt = "A cat sitting on a windowsill, watercolor style"
image = pipe(
    prompt=prompt,
    height=1024,
    width=768,
    guidance_scale=3.5,
    num_inference_steps=50,
).images[0]
image.save("output_lora.png")
```
