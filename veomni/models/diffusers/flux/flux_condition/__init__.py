from ....loader import MODEL_CONFIG_REGISTRY, MODELING_REGISTRY


@MODEL_CONFIG_REGISTRY.register("FluxConditionModel")
def register_flux_condition_config():
    from .configuration_flux_condition import FluxConditionModelConfig

    return FluxConditionModelConfig


@MODELING_REGISTRY.register("FluxConditionModel")
def register_flux_condition_modeling(architecture: str = None):
    from .modeling_flux_condition import FluxConditionModel

    return FluxConditionModel
