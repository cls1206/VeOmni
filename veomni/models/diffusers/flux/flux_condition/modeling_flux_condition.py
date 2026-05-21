from __future__ import annotations

from typing import Any

import torch
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from torchvision.transforms import InterpolationMode, functional
from transformers import CLIPTokenizer, PreTrainedModel, T5EncoderModel

from .....distributed.parallel_state import get_parallel_state
from .....utils import logging
from .....utils.device import get_device_type
from .configuration_flux_condition import FluxConditionModelConfig


logger = logging.get_logger(__name__)


class FluxConditionModel(PreTrainedModel):
    config_class = FluxConditionModelConfig
    supports_gradient_checkpointing = False

    def __init__(self, config: FluxConditionModelConfig, meta_init=False, **kwargs):
        super().__init__(config, **kwargs)
        self.config = config
        self.tokenizer = None
        self.tokenizer_2 = None
        self.text_encoder = None
        self.text_encoder_2 = None
        self.vae = None
        self.scheduler = None
        self._timesteps_ready = False
        self._timesteps_image_seq_len: int | None = None
        self.meta_init = meta_init
        self.seed = config.seed
        self.generator = torch.Generator(device=torch.device(get_device_type()))
        self.generator.manual_seed((self.seed or 0) + get_parallel_state().dp_rank)
        self._load_components()

    @property
    def _execution_device(self):
        if self.vae is not None:
            return self.vae.device
        if self.text_encoder_2 is not None:
            return self.text_encoder_2.device
        return torch.device(get_device_type())

    def _load_components(self):
        base = self.config.base_model_path
        logger.info_rank0(f"Loading Flux condition components from {base}.")
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            base,
            subfolder=self.config.scheduler_subfolder,
        )
        if self.meta_init:
            return

        self.tokenizer = CLIPTokenizer.from_pretrained(base, subfolder=self.config.tokenizer_subfolder)
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(base, subfolder=self.config.tokenizer_2_subfolder)
        from transformers import CLIPTextModel

        self.text_encoder = CLIPTextModel.from_pretrained(
            base,
            subfolder=self.config.text_encoder_subfolder,
            torch_dtype=torch.bfloat16,
        )
        self.text_encoder_2 = T5EncoderModel.from_pretrained(
            base,
            subfolder=self.config.text_encoder_2_subfolder,
            torch_dtype=torch.bfloat16,
        )
        self.vae = AutoencoderKL.from_pretrained(
            base,
            subfolder=self.config.vae_subfolder,
            torch_dtype=torch.float32,
        )

    @staticmethod
    def _calculate_shift(
        image_seq_len: int,
        base_seq_len: int = 256,
        max_seq_len: int = 4096,
        base_shift: float = 0.5,
        max_shift: float = 1.15,
    ) -> float:
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        return image_seq_len * m + b

    @staticmethod
    def _as_list(value: Any, length: int | None = None) -> list[Any]:
        if value is None:
            if length is None:
                return []
            return [None] * length
        if isinstance(value, list):
            return value
        return [value]

    @torch.no_grad()
    def _encode_prompt_using_clip(self, prompt: list[str], device: torch.device, dtype: torch.dtype):
        input_ids = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            max_length=77,
            truncation=True,
        ).input_ids.to(device)
        pooled_prompt_emb = self.text_encoder(input_ids, output_hidden_states=False).pooler_output
        return pooled_prompt_emb

    @torch.no_grad()
    def _encode_prompt_using_t5(self, prompt: list[str], device: torch.device, dtype: torch.dtype):
        input_ids = self.tokenizer_2(
            prompt,
            return_tensors="pt",
            padding="max_length",
            max_length=self.config.max_sequence_length,
            truncation=True,
        ).input_ids.to(device)
        prompt_emb = self.text_encoder_2(input_ids).last_hidden_state
        return prompt_emb

    def _image_to_tensor(self, image) -> torch.Tensor:
        image = image.convert("RGB")
        image = functional.resize(
            image,
            [self.config.height, self.config.width],
            interpolation=InterpolationMode.BICUBIC,
        )
        image = functional.to_tensor(image)
        return image.unsqueeze(0).mul(2.0).sub(1.0)

    def _encode_image_to_latents(self, image) -> torch.Tensor:
        image_tensor = self._image_to_tensor(image).to(device=self.vae.device, dtype=self.vae.dtype)
        posterior: DiagonalGaussianDistribution = self.vae.encode(image_tensor).latent_dist
        return posterior.parameters

    @torch.no_grad()
    def get_condition(self, inputs, images, **kwargs) -> dict[str, Any]:
        prompts = inputs if isinstance(inputs, list) else [inputs]
        device = self._execution_device
        dtype = self.text_encoder_2.dtype if self.text_encoder_2 is not None else torch.bfloat16

        prompt_emb = self._encode_prompt_using_t5(prompts, device, dtype)
        pooled_prompt_emb = self._encode_prompt_using_clip(prompts, device, dtype)

        context_list = [prompt_emb[idx : idx + 1] for idx in range(len(prompts))]
        pooled_context_list = [pooled_prompt_emb[idx : idx + 1] for idx in range(len(prompts))]

        latents_list = []
        for sample_images in images:
            sample_images = self._as_list(sample_images)
            if len(sample_images) != 1:
                raise ValueError("Flux text-to-image training expects exactly one target image per sample.")
            latents_list.append(self._encode_image_to_latents(sample_images[0]))

        return {
            "latents": latents_list,
            "context": context_list,
            "pooled_context": pooled_context_list,
        }

    @staticmethod
    def _pack_latents(latents: torch.Tensor, batch_size: int, num_channels_latents: int, height: int, width: int):
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)
        return latents

    @staticmethod
    def _prepare_latent_image_ids(batch_size: int, height: int, width: int, device: torch.device, dtype: torch.dtype):
        latent_image_ids = torch.zeros(height // 2, width // 2, 3, device=device, dtype=dtype)
        latent_image_ids[..., 1] = (
            latent_image_ids[..., 1] + torch.arange(height // 2, device=device, dtype=dtype)[:, None]
        )
        latent_image_ids[..., 2] = (
            latent_image_ids[..., 2] + torch.arange(width // 2, device=device, dtype=dtype)[None, :]
        )
        latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape
        latent_image_ids = latent_image_ids[None, :].repeat(batch_size, 1, 1, 1)
        latent_image_ids = latent_image_ids.reshape(
            batch_size, latent_image_id_height * latent_image_id_width, latent_image_id_channels
        )
        return latent_image_ids

    def process_condition(
        self,
        latents=None,
        context=None,
        pooled_context=None,
        **kwargs,
    ) -> dict[str, Any]:
        if "hidden_states" in kwargs and "training_target" in kwargs:
            ready_inputs = dict(kwargs)
            if latents is not None:
                ready_inputs["latents"] = latents
            if context is not None:
                ready_inputs["context"] = context
            if pooled_context is not None:
                ready_inputs["pooled_context"] = pooled_context
            return ready_inputs

        if latents is None or context is None or pooled_context is None:
            raise ValueError("Flux condition processing requires latents, context, and pooled_context.")

        latents_list = self._as_list(latents)
        context_list = self._as_list(context, len(latents_list))
        pooled_context_list = self._as_list(pooled_context, len(latents_list))

        first_params = latents_list[0]
        first_h, first_w = int(first_params.shape[-2]), int(first_params.shape[-1])
        image_seq_len = (first_h // 2) * (first_w // 2)

        if (not self._timesteps_ready) or (self._timesteps_image_seq_len != image_seq_len):
            mu = self._calculate_shift(image_seq_len)
            self.scheduler.set_timesteps(
                self.config.num_train_timesteps,
                device=self.generator.device,
                mu=mu,
            )
            self._timesteps_ready = True
            self._timesteps_image_seq_len = image_seq_len

        packed_conditions: dict[str, list[Any]] = {
            "hidden_states": [],
            "timestep": [],
            "encoder_hidden_states": [],
            "pooled_prompt_emb": [],
            "guidance": [],
            "text_ids": [],
            "img_ids": [],
            "training_target": [],
            "latents": [],
        }

        for sample_latents, sample_context, sample_pooled_context in zip(
            latents_list, context_list, pooled_context_list
        ):
            latents_decoded = DiagonalGaussianDistribution(sample_latents).mode()
            latents_decoded = (latents_decoded - self.vae.config.shift_factor) * self.vae.config.scaling_factor
            latents_decoded = latents_decoded.to(self.generator.device)

            noise = torch.randn(
                latents_decoded.shape,
                dtype=latents_decoded.dtype,
                device=self.generator.device,
                generator=self.generator,
            ).to(self.generator.device)

            timestep_ids = torch.randint(
                0,
                len(self.scheduler.timesteps),
                (latents_decoded.shape[0],),
                device=self.generator.device,
                generator=self.generator,
            ).to(latents_decoded.device)
            timestep = self.scheduler.timesteps[timestep_ids].to(
                device=latents_decoded.device, dtype=latents_decoded.dtype
            )

            noisy_latents = self.scheduler.scale_noise(latents_decoded, timestep, noise)
            training_target = noise - latents_decoded

            batch_size = latents_decoded.shape[0]
            _, num_channels, height, width = latents_decoded.shape

            packed_noisy = self._pack_latents(noisy_latents, batch_size, num_channels, height, width)
            packed_target = self._pack_latents(training_target, batch_size, num_channels, height, width)

            text_ids = torch.zeros(
                batch_size,
                sample_context.shape[1],
                3,
                device=latents_decoded.device,
                dtype=latents_decoded.dtype,
            )
            img_ids = self._prepare_latent_image_ids(
                batch_size, height, width, latents_decoded.device, latents_decoded.dtype
            )
            guidance = torch.full(
                [batch_size],
                self.config.guidance_scale,
                device=latents_decoded.device,
                dtype=torch.float32,
            )

            packed_conditions["hidden_states"].append(packed_noisy)
            packed_conditions["timestep"].append(timestep / 1000)
            packed_conditions["encoder_hidden_states"].append(sample_context.to(latents_decoded.device))
            packed_conditions["pooled_prompt_emb"].append(sample_pooled_context.to(latents_decoded.device))
            packed_conditions["guidance"].append(guidance)
            packed_conditions["text_ids"].append(text_ids)
            packed_conditions["img_ids"].append(img_ids)
            packed_conditions["training_target"].append(packed_target)
            packed_conditions["latents"].append(latents_decoded)

        return packed_conditions
