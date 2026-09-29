from abc import ABCMeta

import modules.util.multi_gpu_util as multi
from modules.model.QwenImage21Model import QwenImage21Model
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.modelSetup.mixin.ModelSetupDebugMixin import ModelSetupDebugMixin
from modules.modelSetup.mixin.ModelSetupDiffusionLossMixin import ModelSetupDiffusionLossMixin
from modules.modelSetup.mixin.ModelSetupFlowMatchingMixin import ModelSetupFlowMatchingMixin
from modules.modelSetup.mixin.ModelSetupNoiseMixin import ModelSetupNoiseMixin
from modules.modelSetup.mixin.ModelSetupText2ImageMixin import ModelSetupText2ImageMixin
from modules.util.checkpointing_util import (
    enable_checkpointing_for_qwen3vl_encoder_layers,
    enable_checkpointing_for_qwen_image_21_transformer,
)
from modules.util.config.TrainConfig import TrainConfig
from modules.util.TrainProgress import TrainProgress

import torch
from torch import Tensor


class BaseQwenImage21Setup(
    BaseModelSetup,
    ModelSetupDiffusionLossMixin,
    ModelSetupDebugMixin,
    ModelSetupNoiseMixin,
    ModelSetupFlowMatchingMixin,
    ModelSetupText2ImageMixin,
    metaclass=ABCMeta,
):
    LAYER_PRESETS = {
        "attn-mlp": ["attn", "img_mlp"],
        "attn-only": ["attn"],
        "blocks": ["transformer_blocks"],
        "full": [],
    }

    def setup_optimizations(self, model: QwenImage21Model, config: TrainConfig):
        super().setup_optimizations(model, config)
        self._setup_model_part(
            model, config, "transformer", config.transformer, enable_checkpointing_for_qwen_image_21_transformer,
        )
        self._setup_model_part(
            model, config, "text_encoder", config.text_encoder,
            enable_checkpointing_for_qwen3vl_encoder_layers, disable_fp16_autocast=True,
        )
        self._setup_model_part(model, config, "vae", config.vae)
        self._set_attention_backend(model.transformer, config.attention_mechanism, mask=True)

    def predict(
            self,
            model: QwenImage21Model,
            batch: dict,
            config: TrainConfig,
            train_progress: TrainProgress,
            *,
            deterministic: bool = False,
    ) -> dict:
        with model.autocast_context:
            batch_seed = 0 if deterministic else train_progress.global_step * multi.world_size() + multi.rank()
            generator = torch.Generator(device=config.train_device)
            generator.manual_seed(batch_seed)

            prompt_embeds = batch["text_encoder_hidden_state"]
            prompt_mask = batch["tokens_mask"].bool()
            image_pad_mask = batch["image_pad_mask"].bool()
            max_length = prompt_mask.sum(dim=1).max().item()
            prompt_embeds = prompt_embeds[:, :max_length]
            prompt_mask = prompt_mask[:, :max_length]
            image_pad_mask = image_pad_mask[:, :max_length]
            if prompt_mask.all():
                prompt_mask = None

            latent_image = batch["latent_image"]
            latent_conditioning_image = batch["latent_conditioning_image"]
            scaled_latent_image = model.scale_latents(latent_image)
            scaled_conditioning_image = model.scale_latents(latent_conditioning_image)
            latent_noise = self._create_noise(scaled_latent_image, config, generator)

            shift = model.calculate_timestep_shift(scaled_latent_image.shape[-2], scaled_latent_image.shape[-1])
            timestep = self._get_timestep_discrete(
                model.noise_scheduler.config["num_train_timesteps"],
                deterministic,
                generator,
                scaled_latent_image.shape[0],
                config,
                shift=shift if config.dynamic_timestep_shifting else config.timestep_shift,
            )
            scaled_noisy_latent_image, sigma = self._add_noise_discrete(
                scaled_latent_image, latent_noise, timestep, model.noise_scheduler.timesteps,
            )

            packed_conditioning = model.pack_latents(scaled_conditioning_image)
            packed_target = model.pack_latents(scaled_noisy_latent_image)
            packed_input = torch.cat([packed_conditioning, packed_target], dim=1)

            target_slots = torch.ones(
                image_pad_mask.shape[0], packed_target.shape[1] // 4,
                dtype=torch.bool, device=image_pad_mask.device,
            )
            transformer_img_mask = torch.cat([image_pad_mask, target_slots], dim=1)
            conditioning_shape = (
                1, scaled_conditioning_image.shape[-2], scaled_conditioning_image.shape[-1],
            )
            target_shape = (1, scaled_latent_image.shape[-2], scaled_latent_image.shape[-1])
            img_shapes = [[conditioning_shape, target_shape] for _ in range(latent_image.shape[0])]

            packed_prediction = model.transformer(
                hidden_states=packed_input.to(dtype=model.train_dtype.torch_dtype()),
                timestep=timestep / 1000,
                encoder_hidden_states=prompt_embeds.to(dtype=model.train_dtype.torch_dtype()),
                encoder_hidden_states_mask=prompt_mask,
                img_shapes=img_shapes,
                img_mask=transformer_img_mask,
                return_dict=True,
            ).sample
            packed_prediction = packed_prediction[:, -packed_target.shape[1]:]
            predicted_flow = model.unpack_latents(
                packed_prediction, scaled_latent_image.shape[-2], scaled_latent_image.shape[-1],
            )
            flow = latent_noise - scaled_latent_image
            model_output_data = {
                "loss_type": "target",
                "timestep": timestep,
                "predicted": predicted_flow,
                "target": flow,
            }

            if config.debug_mode:
                with torch.no_grad():
                    predicted_image = scaled_noisy_latent_image - predicted_flow * sigma
                    self._save_latent("1-noise", latent_noise, config, train_progress)
                    self._save_latent("2-noisy_image", scaled_noisy_latent_image, config, train_progress)
                    self._save_latent("3-predicted_flow", predicted_flow, config, train_progress)
                    self._save_latent("4-flow", flow, config, train_progress)
                    self._save_latent("5-predicted_image", predicted_image, config, train_progress)
                    self._save_latent("6-image", scaled_latent_image, config, train_progress)

        return model_output_data

    def calculate_loss(self, model, batch: dict, data: dict, config: TrainConfig) -> Tensor:
        return self._flow_matching_losses(
            batch=batch, data=data, config=config, train_device=self.train_device,
            sigmas=model.noise_scheduler.sigmas,
        ).mean()

    def prepare_text_caching(self, model: QwenImage21Model, config: TrainConfig):
        model.materialize_only("text_encoder")
        model.eval()
