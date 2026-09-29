import math
from collections.abc import Callable

from modules.model.QwenImage21Model import QwenImage21Model
from modules.modelSampler.BaseModelSampler import BaseModelSampler, ModelSamplerOutput
from modules.util import factory
from modules.util.config.SampleConfig import SampleConfig
from modules.util.enum.FileType import FileType
from modules.util.enum.ModelType import ModelType
from modules.util.image_util import load_image

import torch

from PIL import Image


@factory.register(BaseModelSampler, ModelType.QWEN_IMAGE_21)
class QwenImage21Sampler(BaseModelSampler):
    def __init__(self, train_device, temp_device, model: QwenImage21Model, model_type):
        super().__init__(train_device, temp_device)
        self.model = model
        self.pipeline = model.create_pipeline()

    @torch.no_grad()
    def sample(
            self,
            sample_config: SampleConfig,
            destination: str,
            image_format=None,
            video_format=None,
            audio_format=None,
            on_sample: Callable = lambda _: None,
            on_update_progress: Callable = lambda _current, _total: None,
    ):
        self.model.materialize_only("text_encoder", "transformer", "vae")
        generator = torch.Generator(device=self.train_device)
        generator.seed() if sample_config.random_seed else generator.manual_seed(sample_config.seed)
        height = self.quantize_resolution(sample_config.height, 32)
        width = self.quantize_resolution(sample_config.width, 32)
        conditioning_image = None
        pipeline_kwargs = {}
        if sample_config.conditioning_image_path:
            if not self.model.train_config.custom_conditioning_image:
                raise ValueError(
                    "Qwen Image 2.1 conditioning-image sampling requires custom_conditioning_image to be enabled"
                )
            conditioning_image = load_image(sample_config.conditioning_image_path, convert_mode="RGB")
            if sample_config.resize_conditioning_image:
                conditioning_image = conditioning_image.resize((width, height), Image.Resampling.LANCZOS)
                # QwenImage21Pipeline otherwise normalizes condition images to its default 1024x1024 area.
                # Matching that area to the requested output makes its own pre-encode resize retain width/height.
                pipeline_kwargs["output_resolution"] = round(math.sqrt(width * height))
        with self.model.autocast_context:
            output = self.pipeline(
                prompt=sample_config.prompt,
                image=conditioning_image,
                negative_prompt=sample_config.negative_prompt if sample_config.cfg_scale > 1 else None,
                true_cfg_scale=sample_config.cfg_scale,
                height=height,
                width=width,
                num_inference_steps=sample_config.diffusion_steps,
                generator=generator,
                **pipeline_kwargs,
            ).images[0]
        sampler_output = ModelSamplerOutput(file_type=FileType.IMAGE, data=output)
        self.save_sampler_output(sampler_output, destination, image_format, video_format, audio_format)
        on_sample(sampler_output)
