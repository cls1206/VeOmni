# dit preprocess should not be used for llm or mllms
from ..preprocess import PREPROCESSOR_REGISTRY


@PREPROCESSOR_REGISTRY.register("Tom-and-Jerry-VideoGeneration-Dataset")
def tom_and_jerry_preprocess(conversations, **kwargs):
    prompt = conversations["prompt"]
    outputs = {}
    images = {}
    videos = [conversations["video_bytes"]]
    return prompt, outputs, images, videos


@PREPROCESSOR_REGISTRY.register("X2I-text-to-image")
def x2i_text_to_image_preprocess(conversations, **kwargs):
    prompt = conversations["text"]
    outputs = {}
    images = [conversations["image_bytes"]]
    videos = {}
    return prompt, outputs, images, videos
