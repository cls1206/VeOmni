from typing import Optional

from transformers import PretrainedConfig


class FluxConditionModelConfig(PretrainedConfig):
    model_type = "FluxConditionModel"

    def __init__(
        self,
        base_model_path: str = "",
        text_encoder_subfolder: str = "text_encoder",
        text_encoder_2_subfolder: str = "text_encoder_2",
        vae_subfolder: str = "vae",
        tokenizer_subfolder: str = "tokenizer",
        tokenizer_2_subfolder: str = "tokenizer_2",
        max_sequence_length: int = 512,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
        guidance_scale: float = 3.5,
        cfg_negative_prob: float = 0.0,
        image_max_size: int = 1024,
        seed: Optional[int] = 42,
        **kwargs,
    ):
        self.base_model_path = base_model_path
        self.text_encoder_subfolder = text_encoder_subfolder
        self.text_encoder_2_subfolder = text_encoder_2_subfolder
        self.vae_subfolder = vae_subfolder
        self.tokenizer_subfolder = tokenizer_subfolder
        self.tokenizer_2_subfolder = tokenizer_2_subfolder
        self.max_sequence_length = max_sequence_length
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.guidance_scale = guidance_scale
        self.cfg_negative_prob = cfg_negative_prob
        self.image_max_size = image_max_size
        self.seed = seed
        super().__init__(**kwargs)

    @classmethod
    def get_config_dict(
        cls,
        pretrained_model_name_or_path,
        **kwargs,
    ):
        config_dict, kwargs = super().get_config_dict(pretrained_model_name_or_path, **kwargs)
        config_dict["base_model_path"] = pretrained_model_name_or_path
        return config_dict, kwargs
