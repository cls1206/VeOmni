from ....loader import MODEL_CONFIG_REGISTRY, MODELING_REGISTRY


@MODEL_CONFIG_REGISTRY.register("FluxTransformer2DModel")
def register_flux_transformer_config():
    from .configuration_flux_transformer import FluxTransformer2DModelConfig

    return FluxTransformer2DModelConfig


@MODELING_REGISTRY.register("FluxTransformer2DModel")
def register_flux_transformer_modeling(architecture: str = None):
    from .modeling_flux_transformer import FluxTransformer2DModel as VeOmniFluxTransformer2DModel

    return VeOmniFluxTransformer2DModel
