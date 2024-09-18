# coding=utf-8
""" mPLUGOwl3 model configuration"""

import os
from typing import Union

from transformers.utils import logging
from configuration_hyper_qwen2 import HyperQwen2Config
from transformers.models.siglip.configuration_siglip import SiglipVisionConfig
logger = logging.get_logger(__name__)


class mPLUGOwl3Config(HyperQwen2Config):
    model_type = "mplugowl3"
    keys_to_ignore_at_inference = ["past_key_values"]

    default_vision_config = {
        "hidden_size": 1152,
        "image_size": 384,
        "intermediate_size": 4304,
        "model_type": "siglip_vision_model",
        "num_attention_heads": 16,
        "num_hidden_layers": 27,
        "patch_size": 14
    }


    def __init__(
        self,
        use_cache=True,
        vision_config=None,
        **kwargs,
    ):
        self.use_cache = use_cache

        # same as HuggingFaceM4/siglip-so400m-14-980-flash-attn2-navit add tgt_sizes
        if vision_config is None:
            self.vision_config = SiglipVisionConfig(**self.default_vision_config)
            logger.info("vision_config is None, using default vision config")
        elif isinstance(vision_config, dict):
            self.vision_config = SiglipVisionConfig(**vision_config)
        elif isinstance(vision_config, SiglipVisionConfig):
            self.vision_config = vision_config
        self.image_size = self.vision_config.image_size
        self.patch_size = self.vision_config.patch_size

        super().__init__(**kwargs)
