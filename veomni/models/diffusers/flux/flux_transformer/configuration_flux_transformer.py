import inspect
from typing import Optional, Tuple

import diffusers
from diffusers import FluxTransformer2DModel
from transformers import PretrainedConfig


FLUX_INIT_SIGNATURE = inspect.signature(FluxTransformer2DModel.__init__)
diffusers_version = diffusers.__version__


class FluxTransformer2DModelConfig(PretrainedConfig):
    model_type = "FluxTransformer2DModel"
    condition_model_type = "FluxConditionModel"

    def __init__(
        self,
        patch_size: int = 1,
        in_channels: int = 64,
        out_channels: Optional[int] = None,
        hidden_size: int = 3072,
        num_attention_heads: int = 24,
        attention_head_dim: int = 128,
        num_layers: int = 19,
        num_single_layers: int = 38,
        joint_attention_dim: int = 4096,
        pooled_projection_dim: int = 768,
        guidance_embeds: bool = True,
        axes_dims_rope: Tuple[int, ...] = (16, 56, 56),
        **kwargs,
    ):
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.num_layers = num_layers
        self.num_single_layers = num_single_layers
        self.joint_attention_dim = joint_attention_dim
        self.pooled_projection_dim = pooled_projection_dim
        self.guidance_embeds = guidance_embeds
        self.axes_dims_rope = axes_dims_rope
        super().__init__(**kwargs)

    def to_diffuser_dict(self):
        return {key: getattr(self, key) for key in FLUX_INIT_SIGNATURE.parameters.keys() if key != "self"}

    def to_dict(self):
        return_dict = super().to_dict()
        return_dict["_class_name"] = "FluxTransformer2DModel"
        return_dict["_diffusers_version"] = diffusers_version
        return_dict.pop("dtype", None)
        return return_dict
