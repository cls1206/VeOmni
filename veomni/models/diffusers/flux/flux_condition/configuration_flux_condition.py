from typing import Optional

from transformers import PretrainedConfig


class FluxConditionModelConfig(PretrainedConfig):
    model_type = "FluxConditionModel"

    def __init__(
        self,
        base_model_path: str = "",
        tokenizer_subfolder: str = "tokenizer",
        tokenizer_2_subfolder: str = "tokenizer_2",
        text_encoder_subfolder: str = "text_encoder",
        text_encoder_2_subfolder: str = "text_encoder_2",
        vae_subfolder: str = "vae",
        scheduler_subfolder: str = "scheduler",
        max_sequence_length: int = 512,
        num_train_timesteps: int = 1000,
        shift: float = 1.15,
        height: int = 1024,
        width: int = 768,
        guidance_scale: float = 3.5,
        seed: Optional[int] = 42,
        **kwargs,
    ):
        self.base_model_path = base_model_path
        self.tokenizer_subfolder = tokenizer_subfolder
        self.tokenizer_2_subfolder = tokenizer_2_subfolder
        self.text_encoder_subfolder = text_encoder_subfolder
        self.text_encoder_2_subfolder = text_encoder_2_subfolder
        self.vae_subfolder = vae_subfolder
        self.scheduler_subfolder = scheduler_subfolder
        self.max_sequence_length = max_sequence_length
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.height = height
        self.width = width
        self.guidance_scale = guidance_scale
        self.seed = seed
        super().__init__(**kwargs)

    @classmethod
    def get_config_dict(
        cls,
        pretrained_model_name_or_path,
        **kwargs,
    ):
        try:
            config_dict, kwargs = super().get_config_dict(pretrained_model_name_or_path, **kwargs)
        except Exception:
            config_dict = {}
        config_dict["base_model_path"] = pretrained_model_name_or_path
        return config_dict, kwargs
