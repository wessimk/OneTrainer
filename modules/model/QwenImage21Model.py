import math
from contextlib import nullcontext

from modules.model.BaseModel import BaseModel
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util.enum.ModelFormat import ModelFormat
from modules.util.enum.ModelType import ModelType
from modules.util.LayerOffloadConductor import LayerOffloadConductor

import torch
import torch.nn.functional as F
from torch import Tensor

from diffusers import (
    AutoencoderKLQwenImage21,
    DiffusionPipeline,
    FlowMatchEulerDiscreteScheduler,
    QwenImage21Pipeline,
    QwenImage21Transformer2DModel,
)
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

PROMPT_MAX_LENGTH = 2048
SYSTEM_PROMPT = "Comprehend and analyze the provided prompt."


class QwenImage21Model(BaseModel):
    processor: Qwen3VLProcessor | None
    noise_scheduler: FlowMatchEulerDiscreteScheduler | None
    text_encoder: Qwen3VLForConditionalGeneration | None
    vae: AutoencoderKLQwenImage21 | None
    transformer: QwenImage21Transformer2DModel | None

    text_encoder_autocast_context: torch.autocast | nullcontext
    text_encoder_offload_conductor: LayerOffloadConductor | None
    transformer_offload_conductor: LayerOffloadConductor | None

    text_encoder_lora: LoRAModuleWrapper | None
    transformer_lora: LoRAModuleWrapper | None
    lora_state_dict: dict | None

    def __init__(self, model_type: ModelType):
        super().__init__(model_type=model_type)
        self.processor = None
        self.noise_scheduler = None
        self.text_encoder = None
        self.vae = None
        self.transformer = None
        self.text_encoder_autocast_context = nullcontext()
        self.text_encoder_offload_conductor = None
        self.transformer_offload_conductor = None
        self.text_encoder_lora = None
        self.transformer_lora = None
        self.lora_state_dict = None
        self._drop_idx = None
        self._img_token_id = None

    @property
    def tokenizer(self):
        return self.processor.tokenizer if self.processor is not None else None

    def initialize_prompt_metadata(self):
        system_message = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}]
        system_tokens = self.processor.apply_chat_template(system_message, tokenize=True, return_dict=False)
        self._drop_idx = len(system_tokens[0])
        self._img_token_id = self.processor.tokenizer.encode("<|image_pad|>")[0]

    def lora_text_encoders(self) -> list[tuple[torch.nn.Module | None, dict[ModelFormat, str]]]:
        return [
            (self.text_encoder, {
                ModelFormat.DIFFUSERS_LORA: "text_encoder",
                ModelFormat.KOHYA_LORA: "lora_te",
                ModelFormat.COMFY_LORA: "text_encoders.qwen3_vl.transformer",
            }),
        ]

    def create_pipeline(self) -> DiffusionPipeline:
        return QwenImage21Pipeline(
            transformer=self.transformer,
            scheduler=self.noise_scheduler,
            vae=self.vae,
            text_encoder=self.text_encoder,
            processor=self.processor,
        )

    def encode_text(
            self,
            text: str | list[str],
            conditioning_images: list | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if isinstance(text, str):
            text = [text]
        text = [prompt if prompt else " " for prompt in text]
        is_t2i = conditioning_images is None

        if is_t2i:
            prompts = [
                f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
                f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
                for prompt in text
            ]
        else:
            if len(conditioning_images) != len(text):
                raise ValueError("Qwen Image 2.1 expects one conditioning image per prompt during training")
            prompts = [
                f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
                f"<|im_start|>user\n<image1><|vision_start|><|image_pad|><|vision_end|>{prompt}<|im_end|>\n"
                f"<|im_start|>assistant\n"
                for prompt in text
            ]

        processor_kwargs = {
            "text": prompts,
            "padding": True,
            "padding_side": "left",
            "return_tensors": "pt",
        }
        if not is_t2i:
            processor_kwargs["images"] = conditioning_images
        model_inputs = self.processor(**processor_kwargs).to(self.text_encoder.device)

        forward_kwargs = {
            "input_ids": model_inputs.input_ids,
            "attention_mask": model_inputs.attention_mask,
            "output_hidden_states": True,
            "use_cache": False,
        }
        if hasattr(model_inputs, "pixel_values"):
            forward_kwargs.update(
                pixel_values=model_inputs.pixel_values,
                image_grid_thw=model_inputs.image_grid_thw,
            )
        if hasattr(model_inputs, "mm_token_type_ids"):
            forward_kwargs["mm_token_type_ids"] = model_inputs.mm_token_type_ids

        text_model = getattr(self.text_encoder.model, "language_model", self.text_encoder.model)
        handle = text_model.norm.register_forward_hook(lambda _module, args, _output: args[0])
        try:
            with self.text_encoder_autocast_context:
                outputs = self.text_encoder(**forward_kwargs)
        finally:
            handle.remove()

        hidden_states = outputs.hidden_states[-1]
        split_hidden_states = []
        split_image_masks = []
        for states, ids, mask in zip(
                hidden_states, model_inputs.input_ids, model_inputs.attention_mask, strict=True,
        ):
            valid = mask.bool()
            split_hidden_states.append(states[valid][self._drop_idx:])
            split_image_masks.append((ids[valid] == self._img_token_id)[self._drop_idx:])

        max_length = max(states.shape[0] for states in split_hidden_states)
        if max_length > PROMPT_MAX_LENGTH:
            raise ValueError(
                f"Qwen Image 2.1 prompt and vision tokens require {max_length} positions, "
                f"which exceeds OneTrainer's limit of {PROMPT_MAX_LENGTH}"
            )
        prompt_embeds = torch.stack([
            F.pad(states, (0, 0, 0, PROMPT_MAX_LENGTH - states.shape[0]))
            for states in split_hidden_states
        ])
        attention_mask = torch.stack([
            F.pad(torch.ones(states.shape[0], dtype=torch.bool, device=states.device),
                  (0, PROMPT_MAX_LENGTH - states.shape[0]))
            for states in split_hidden_states
        ])
        image_pad_mask = torch.stack([
            F.pad(image_mask, (0, PROMPT_MAX_LENGTH - image_mask.shape[0]))
            for image_mask in split_image_masks
        ])

        return prompt_embeds, attention_mask, image_pad_mask

    @staticmethod
    def pack_latents(latents: Tensor) -> Tensor:
        batch_size, channels, frames, height, width = latents.shape
        if frames != 1:
            raise ValueError("Qwen Image 2.1 training currently supports still images only")
        return latents.reshape(batch_size, channels, height * width).transpose(1, 2)

    @staticmethod
    def unpack_latents(latents: Tensor, height: int, width: int) -> Tensor:
        batch_size, _, channels = latents.shape
        return latents.transpose(1, 2).reshape(batch_size, channels, 1, height, width)

    def scale_latents(self, latents: Tensor) -> Tensor:
        mean = torch.tensor(self.vae.config.latents_mean, device=latents.device, dtype=latents.dtype).view(1, -1, 1, 1, 1)
        std = torch.tensor(self.vae.config.latents_std, device=latents.device, dtype=latents.dtype).view(1, -1, 1, 1, 1)
        return (latents - mean) / std

    def unscale_latents(self, latents: Tensor) -> Tensor:
        mean = torch.tensor(self.vae.config.latents_mean, device=latents.device, dtype=latents.dtype).view(1, -1, 1, 1, 1)
        std = torch.tensor(self.vae.config.latents_std, device=latents.device, dtype=latents.dtype).view(1, -1, 1, 1, 1)
        return latents * std + mean

    def calculate_timestep_shift(self, latent_height: int, latent_width: int) -> float:
        image_seq_len = latent_height * latent_width
        base_seq_len = self.noise_scheduler.config.base_image_seq_len
        max_seq_len = self.noise_scheduler.config.max_image_seq_len
        base_shift = self.noise_scheduler.config.base_shift
        max_shift = self.noise_scheduler.config.max_shift
        mu = base_shift + (max_shift - base_shift) * (image_seq_len - base_seq_len) / (max_seq_len - base_seq_len)
        return math.exp(mu)
