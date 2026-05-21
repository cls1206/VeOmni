from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from diffusers import FluxTransformer2DModel as _FluxTransformer2DModel
from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from .configuration_flux_transformer import FLUX_INIT_SIGNATURE, FluxTransformer2DModelConfig


@dataclass
class FluxModelOutput(ModelOutput):
    loss: dict[str, torch.FloatTensor] | None = None
    predictions: list[torch.FloatTensor] | None = None


class _FluxTransformerInitShim(_FluxTransformer2DModel):
    def __init__(self, *args, **kwargs):
        torch.nn.Module.__init__(self)


class FluxTransformer2DModel(PreTrainedModel, _FluxTransformerInitShim):
    config_class = FluxTransformer2DModelConfig
    supports_gradient_checkpointing = True
    _no_split_modules = ["FluxTransformerBlock", "FluxSingleTransformerBlock"]

    def __init__(self, config: FluxTransformer2DModelConfig, **kwargs):
        PreTrainedModel.__init__(self, config, **kwargs)
        if hasattr(self, "_internal_dict"):
            del self._internal_dict
        kwargs.pop("attn_implementation", None)
        kwargs.pop("torch_dtype", None)
        _FluxTransformer2DModel.__init__(self, **config.to_diffuser_dict())
        self.config: FluxTransformer2DModelConfig = config
        self.config.tie_word_embeddings = False

    @property
    def config(self):
        return self._internal_dict

    @config.setter
    def config(self, value):
        self._internal_dict = value

    @staticmethod
    def _as_list(value: Any, length: int | None = None) -> list[Any]:
        if value is None:
            if length is None:
                return []
            return [None] * length
        if isinstance(value, list):
            return value
        if length is not None and isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == length:
            return [value[idx : idx + 1] for idx in range(length)]
        return [value]

    def forward(
        self,
        hidden_states: torch.Tensor | list[torch.Tensor],
        timestep: torch.Tensor | list[torch.Tensor],
        encoder_hidden_states: torch.Tensor | list[torch.Tensor],
        pooled_prompt_emb: torch.Tensor | list[torch.Tensor],
        guidance: torch.Tensor | list[torch.Tensor],
        text_ids: torch.Tensor | list[torch.Tensor],
        img_ids: torch.Tensor | list[torch.Tensor],
        training_target: torch.Tensor | list[torch.Tensor],
        latents: torch.Tensor | list[torch.Tensor] | None = None,
    ):
        hidden_states_list = self._as_list(hidden_states)
        sample_count = len(hidden_states_list)
        timestep_list = self._as_list(timestep, sample_count)
        encoder_hidden_states_list = self._as_list(encoder_hidden_states, sample_count)
        pooled_prompt_emb_list = self._as_list(pooled_prompt_emb, sample_count)
        guidance_list = self._as_list(guidance, sample_count)
        text_ids_list = self._as_list(text_ids, sample_count)
        img_ids_list = self._as_list(img_ids, sample_count)
        target_list = self._as_list(training_target, sample_count)

        param_dtype = self.dtype

        per_sample_losses = []
        predictions = []
        for (
            sample_hs,
            sample_ts,
            sample_enc_hs,
            sample_ppe,
            sample_g,
            sample_ti,
            sample_img_ids,
            sample_target,
        ) in zip(
            hidden_states_list,
            timestep_list,
            encoder_hidden_states_list,
            pooled_prompt_emb_list,
            guidance_list,
            text_ids_list,
            img_ids_list,
            target_list,
        ):
            sample_hs = sample_hs.to(dtype=param_dtype)
            sample_enc_hs = sample_enc_hs.to(dtype=param_dtype)
            sample_ppe = sample_ppe.to(dtype=param_dtype)

            prediction = _FluxTransformer2DModel.forward(
                self,
                hidden_states=sample_hs,
                timestep=sample_ts,
                encoder_hidden_states=sample_enc_hs,
                pooled_projections=sample_ppe,
                guidance=sample_g,
                text_ids=sample_ti,
                img_ids=sample_img_ids,
                return_dict=False,
            )[0]
            predictions.append(prediction)
            per_sample_loss = F.mse_loss(prediction.float(), sample_target.float(), reduction="none")
            per_sample_loss = per_sample_loss.view(per_sample_loss.shape[0], -1).mean(dim=1)
            per_sample_losses.append(per_sample_loss)

        loss = torch.stack(per_sample_losses).mean()
        return FluxModelOutput(loss={"mse_loss": loss}, predictions=predictions)

    def save_pretrained(self, path, **kwargs):
        hf_config = copy.deepcopy(self.config)
        self.config = self.config.to_diffuser_dict()
        _FluxTransformer2DModel.save_pretrained(self, path, **kwargs)
        self.config = hf_config

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        diffusers_model = _FluxTransformer2DModel.from_pretrained(path, **kwargs)
        diffusers_model.__class__ = cls

        valid_keys = set(FLUX_INIT_SIGNATURE.parameters) - {"self"}
        diffusers_cfg = dict(diffusers_model.config)
        veomni_cfg = cls.config_class(**{k: v for k, v in diffusers_cfg.items() if k in valid_keys})
        diffusers_model.config = veomni_cfg
        diffusers_model.config.tie_word_embeddings = False
        return diffusers_model
