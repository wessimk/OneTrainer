import os

from modules.dataLoader.BaseDataLoader import BaseDataLoader
from modules.dataLoader.mixin.DataLoaderText2ImageMixin import DataLoaderText2ImageMixin
from modules.model.QwenImage21Model import QwenImage21Model
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.modelSetup.BaseQwenImage21Setup import BaseQwenImage21Setup
from modules.util import factory
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType
from modules.util.TrainProgress import TrainProgress

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModules.DecodeVAE import DecodeVAE
from mgds.pipelineModules.EncodeVAE import EncodeVAE
from mgds.pipelineModules.ImageToVideo import ImageToVideo
from mgds.pipelineModules.RescaleImageChannels import RescaleImageChannels
from mgds.pipelineModules.SampleVAEDistribution import SampleVAEDistribution
from mgds.pipelineModules.SaveImage import SaveImage
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule

import torch


class EncodeQwenImage21Prompt(PipelineModule, RandomAccessPipelineModule):
    def __init__(self, model: QwenImage21Model):
        super().__init__()
        self.model = model

    def length(self):
        return self._get_previous_length("prompt")

    def get_inputs(self):
        return ["prompt", "conditioning_image"]

    def get_outputs(self):
        return ["text_encoder_hidden_state", "tokens_mask", "image_pad_mask"]

    def get_item(self, variation: int, index: int, requested_name: str = None):
        prompt = self._get_previous_item(variation, "prompt", index)
        conditioning_image = self._get_previous_item(variation, "conditioning_image", index)
        hidden_state, attention_mask, image_pad_mask = self.model.encode_text(
            [prompt], conditioning_images=[conditioning_image],
        )
        return {
            "text_encoder_hidden_state": hidden_state.squeeze(0),
            "tokens_mask": attention_mask.squeeze(0),
            "image_pad_mask": image_pad_mask.squeeze(0),
        }


class AddQwenImage21Alpha(PipelineModule, RandomAccessPipelineModule):
    """Convert OneTrainer's RGB image tensors to the RGBA input expected by the Qwen Image 2.1 VAE."""

    def __init__(self, in_name: str, out_name: str):
        super().__init__()
        self.in_name = in_name
        self.out_name = out_name

    def length(self):
        return self._get_previous_length(self.in_name)

    def get_inputs(self):
        return [self.in_name]

    def get_outputs(self):
        return [self.out_name]

    def get_item(self, variation: int, index: int, requested_name: str = None):
        image = self._get_previous_item(variation, self.in_name, index)
        if image.shape[0] == 3:
            image = torch.cat([image, torch.ones_like(image[:1])], dim=0)
        elif image.shape[0] != 4:
            raise ValueError(f"Qwen Image 2.1 VAE expects RGB or RGBA input, received {image.shape[0]} channels")
        return {self.out_name: image}


@factory.register(BaseDataLoader, ModelType.QWEN_IMAGE_21)
class QwenImage21BaseDataLoader(BaseDataLoader, DataLoaderText2ImageMixin):
    def _preparation_modules(self, config: TrainConfig, model: QwenImage21Model):
        if config.text_encoder.train:
            raise NotImplementedError("Qwen Image 2.1 text-encoder training is not supported yet")

        encode_prompt = EncodeQwenImage21Prompt(model)
        rescale_image = RescaleImageChannels(
            image_in_name="image", image_out_name="image",
            in_range_min=0, in_range_max=1, out_range_min=-1, out_range_max=1,
        )
        rescale_condition = RescaleImageChannels(
            image_in_name="conditioning_image", image_out_name="conditioning_image",
            in_range_min=0, in_range_max=1, out_range_min=-1, out_range_max=1,
        )
        add_image_alpha = AddQwenImage21Alpha(in_name="image", out_name="image")
        add_condition_alpha = AddQwenImage21Alpha(
            in_name="conditioning_image", out_name="conditioning_image",
        )
        condition_to_video = ImageToVideo(in_name="conditioning_image", out_name="conditioning_image")
        encode_image = EncodeVAE(
            in_name="image", out_name="latent_image_distribution", vae=model.vae,
            autocast_contexts=[model.autocast_context], dtype=model.train_dtype.torch_dtype(), sample_mode="argmax",
        )
        encode_condition = EncodeVAE(
            in_name="conditioning_image", out_name="latent_conditioning_distribution", vae=model.vae,
            autocast_contexts=[model.autocast_context], dtype=model.train_dtype.torch_dtype(), sample_mode="argmax",
        )
        sample_image = SampleVAEDistribution(
            in_name="latent_image_distribution", out_name="latent_image", mode="mean",
        )
        sample_condition = SampleVAEDistribution(
            in_name="latent_conditioning_distribution", out_name="latent_conditioning_image", mode="mean",
        )
        return [
            encode_prompt,
            rescale_image, rescale_condition, add_image_alpha, add_condition_alpha, condition_to_video,
            encode_image, encode_condition, sample_image, sample_condition,
        ]

    def _cache_modules(self, config, model, model_setup: BaseQwenImage21Setup):
        def prepare_conditioning_cache():
            model.materialize_only("text_encoder", "vae")
            model.eval()

        image_split_names = [
            "latent_image", "latent_conditioning_image",
            "text_encoder_hidden_state", "tokens_mask", "image_pad_mask",
            "original_resolution", "crop_offset",
        ]
        image_aggregate_names = ["crop_resolution", "image_path"]
        sort_names = image_aggregate_names + image_split_names + ["prompt", "concept"]
        return self._cache_modules_from_names(
            model, model_setup,
            image_split_names=image_split_names,
            image_aggregate_names=image_aggregate_names,
            text_split_names=[],
            sort_names=sort_names,
            config=config,
            text_caching=False,
            before_cache_image_fun=prepare_conditioning_cache,
        )

    def _output_modules(self, config, model, model_setup):
        return self._output_modules_from_out_names(
            model, model_setup,
            output_names=[
                "image_path", "latent_image", "latent_conditioning_image", "prompt",
                "text_encoder_hidden_state", "tokens_mask", "image_pad_mask",
                "original_resolution", "crop_resolution", "crop_offset",
            ],
            config=config,
            use_conditioning_image=False,
            vae=model.vae,
            autocast_context=[model.autocast_context],
            train_dtype=model.train_dtype,
        )

    def _debug_modules(self, config, model):
        debug_dir = os.path.join(config.debug_dir, "dataloader")

        def prepare_vae():
            model.materialize("vae")

        decode_image = DecodeVAE(
            in_name="latent_image", out_name="decoded_image", vae=model.vae,
            autocast_contexts=[model.autocast_context], dtype=model.train_dtype.torch_dtype(),
        )
        decode_condition = DecodeVAE(
            in_name="latent_conditioning_image", out_name="decoded_conditioning_image", vae=model.vae,
            autocast_contexts=[model.autocast_context], dtype=model.train_dtype.torch_dtype(),
        )
        save_image = SaveImage(
            image_in_name="decoded_image", original_path_in_name="image_path", path=debug_dir,
            in_range_min=-1, in_range_max=1, before_save_fun=prepare_vae,
        )
        save_condition = SaveImage(
            image_in_name="decoded_conditioning_image", original_path_in_name="image_path", path=debug_dir,
            in_range_min=-1, in_range_max=1, before_save_fun=prepare_vae,
        )
        return [decode_image, decode_condition, save_image, save_condition]

    def _create_dataset(
            self,
            config: TrainConfig,
            model: QwenImage21Model,
            model_setup: BaseModelSetup,
            train_progress: TrainProgress,
            is_validation: bool = False,
    ):
        if not config.custom_conditioning_image:
            raise ValueError(
                "Qwen Image 2.1 training currently requires custom_conditioning_image and paired *-condlabel images"
            )
        return DataLoaderText2ImageMixin._create_dataset(
            self, config, model, model_setup, train_progress, is_validation,
            aspect_bucketing_quantization=32,
            allow_video_files=False,
            vae_frame_dim=True,
            supports_inpainting=True,
        )
