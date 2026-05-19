from __future__ import annotations

from typing import Any

import torch
from diffusers import AutoencoderKL
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
from transformers import AutoTokenizer, CLIPTextModel, PreTrainedModel, T5EncoderModel

from .....distributed.parallel_state import get_parallel_state
from .....schedulers.flow_match import FlowMatchScheduler
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
        self.negative_prompt_embeds = None
        self.negative_pooled_prompt_embeds = None
        self._timesteps_ready = False
        self.meta_init = meta_init
        self.seed = config.seed
        self.generator = torch.Generator(device=torch.device(get_device_type()))
        self.generator.manual_seed(self.seed + get_parallel_state().dp_rank)
        self._load_components()

    @property
    def _execution_device(self):
        return self.vae.device

    def _load_components(self):
        base = self.config.base_model_path
        logger.info_rank0(f"Loading Flux condition components from {base}.")

        self.tokenizer = AutoTokenizer.from_pretrained(base, subfolder=self.config.tokenizer_subfolder)
        self.tokenizer_2 = AutoTokenizer.from_pretrained(base, subfolder=self.config.tokenizer_2_subfolder)

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

        if self.meta_init:
            self.vae = AutoencoderKL.from_config(
                base,
                subfolder=self.config.vae_subfolder,
                torch_dtype=torch.float32,
            )
        else:
            self.vae = AutoencoderKL.from_pretrained(
                base,
                subfolder=self.config.vae_subfolder,
                torch_dtype=torch.float32,
            )

        self.scheduler = FlowMatchScheduler(
            num_train_timesteps=self.config.num_train_timesteps,
            shift=self.config.shift,
        )

        self._prepare_negative_prompt_embeds()

        if self.meta_init:
            del self.text_encoder
            del self.text_encoder_2

    @torch.no_grad()
    def _prepare_negative_prompt_embeds(self):
        prompt_embeds, pooled_prompt_embeds, _, _ = FluxPipeline.encode_prompt(
            self,
            prompt=[""],
            prompt_2=[""],
            do_classifier_free_guidance=False,
            num_images_per_prompt=1,
            device=self.text_encoder.device,
        )
        self.negative_prompt_embeds = prompt_embeds[0].unsqueeze(0)
        self.negative_pooled_prompt_embeds = pooled_prompt_embeds[0].unsqueeze(0)

    @torch.no_grad()
    def _encode_prompt(self, prompts: list[str]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        prompt_embeds, pooled_prompt_embeds, _, _ = FluxPipeline.encode_prompt(
            self,
            prompt=prompts,
            prompt_2=prompts,
            do_classifier_free_guidance=False,
            num_images_per_prompt=1,
            device=self.text_encoder.device,
            max_sequence_length=self.config.max_sequence_length,
        )
        context_list = [u.unsqueeze(0) for u in prompt_embeds]
        pooled_context_list = [u.unsqueeze(0) for u in pooled_prompt_embeds]
        return context_list, pooled_context_list

    @torch.no_grad()
    def _encode_image_to_latents(self, images: list) -> list[torch.Tensor]:
        import numpy as np
        from PIL import Image
        from torchvision.transforms import InterpolationMode, functional

        latents_list = []
        for sample_images in images:
            assert len(sample_images) == 1, "Only one image per sample is supported"
            image = sample_images[0]

            if isinstance(image, Image.Image):
                image = np.array(image).astype(np.float32) / 255.0
                image = torch.from_numpy(image).permute(2, 0, 1)
            elif isinstance(image, torch.Tensor):
                if image.dim() == 4:
                    image = image.squeeze(0)
                if image.max() > 1.0:
                    image = image.float() / 255.0
            else:
                raise ValueError(f"Unsupported image type: {type(image)}")

            height, width = image.shape[-2:]
            size = min(self.config.image_max_size, min(width, height))
            image = functional.resize(image, size, interpolation=InterpolationMode.BICUBIC)
            image = image.clamp(0.0, 1.0) * 2.0 - 1.0

            if image.dim() == 3:
                image = image.unsqueeze(0)

            image = image.to(device=self.vae.device, dtype=self.vae.dtype)
            posterior = self.vae.encode(image).latent_dist
            latents_list.append(posterior.parameters)

        return latents_list

    @torch.no_grad()
    def get_condition(self, inputs, images=None, videos=None, **kwargs) -> dict[str, Any]:
        """
        inputs: list[str], a list of prompts
        images: list[list[torch.Tensor]], a list of samples of images
        videos: not used for Flux (image-only model)
        """
        context_list, pooled_context_list = self._encode_prompt(inputs)

        if images is not None and len(images) > 0 and any(img for img in images):
            latents_list = self._encode_image_to_latents(images)
        else:
            raise ValueError("FluxConditionModel requires images input. Videos are not supported.")

        return {
            "latents": latents_list,
            "context": context_list,
            "pooled_context": pooled_context_list,
        }

    def process_condition(
        self,
        latents: list[torch.Tensor],
        context: list[torch.Tensor],
        pooled_context: list[torch.Tensor],
    ) -> dict[str, Any]:
        if not self._timesteps_ready:
            self.scheduler.set_timesteps(self.config.num_train_timesteps, training=True)
            self._timesteps_ready = True

        packed_conditions: dict[str, list[torch.Tensor]] = {
            "hidden_states": [],
            "timestep": [],
            "encoder_hidden_states": [],
            "pooled_prompt_emb": [],
            "guidance": [],
            "text_ids": [],
            "training_target": [],
            "latents": [],
        }

        for sample_latents, sample_context, sample_pooled_context in zip(latents, context, pooled_context):
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
                self.scheduler.num_train_timesteps,
                (latents_decoded.shape[0],),
                device=self.generator.device,
                generator=self.generator,
            ).to(latents_decoded.device)
            timestep = self.scheduler.timesteps[timestep_ids].to(
                device=latents_decoded.device, dtype=latents_decoded.dtype
            )

            noisy_latents = self.scheduler.add_noise(latents_decoded, noise, timestep, latents_decoded.shape[0], False)
            training_target = self.scheduler.training_target(latents_decoded, noise, timestep)

            use_negative_context = (
                self.config.cfg_negative_prob > 0
                and torch.rand((), device=self.generator.device, generator=self.generator)
                < self.config.cfg_negative_prob
            )
            if use_negative_context:
                sample_context = self.negative_prompt_embeds.to(
                    device=latents_decoded.device, dtype=sample_context.dtype
                )
                sample_pooled_context = self.negative_pooled_prompt_embeds.to(
                    device=latents_decoded.device, dtype=sample_pooled_context.dtype
                )
            else:
                sample_context = sample_context.to(latents_decoded.device)
                sample_pooled_context = sample_pooled_context.to(latents_decoded.device)

            guidance = torch.full(
                (latents_decoded.shape[0],),
                self.config.guidance_scale,
                device=latents_decoded.device,
                dtype=latents_decoded.dtype,
            )

            batch_size = latents_decoded.shape[0]
            text_ids = torch.zeros(
                batch_size, sample_context.shape[1], 3, device=latents_decoded.device, dtype=latents_decoded.dtype
            )

            packed_conditions["hidden_states"].append(noisy_latents)
            packed_conditions["timestep"].append(timestep)
            packed_conditions["encoder_hidden_states"].append(sample_context)
            packed_conditions["pooled_prompt_emb"].append(sample_pooled_context)
            packed_conditions["guidance"].append(guidance)
            packed_conditions["text_ids"].append(text_ids)
            packed_conditions["training_target"].append(training_target)
            packed_conditions["latents"].append(latents_decoded)

        return packed_conditions
