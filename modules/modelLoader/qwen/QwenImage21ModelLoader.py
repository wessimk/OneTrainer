import os
import traceback

from modules.model.QwenImage21Model import QwenImage21Model
from modules.modelLoader.mixin.HFModelLoaderMixin import HFModelLoaderMixin
from modules.util.config.TrainConfig import QuantizationConfig
from modules.util.enum.ModelType import ModelType
from modules.util.ModelNames import ModelNames
from modules.util.ModelWeightDtypes import ModelWeightDtypes

import torch

from diffusers import (
    AutoencoderKLQwenImage21,
    FlowMatchEulerDiscreteScheduler,
    GGUFQuantizationConfig,
    QwenImage21Transformer2DModel,
)
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor


class QwenImage21ModelLoader(HFModelLoaderMixin):
    def __load_diffusers(
            self,
            model: QwenImage21Model,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            transformer_model_name: str,
            vae_model_name: str,
            quantization: QuantizationConfig,
    ):
        processor = Qwen3VLProcessor.from_pretrained(base_model_name, subfolder="processor")
        noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(base_model_name, subfolder="scheduler")
        text_encoder = self._load_transformers_sub_module(
            Qwen3VLForConditionalGeneration,
            weight_dtypes.text_encoder,
            weight_dtypes.fallback_train_dtype,
            base_model_name,
            "text_encoder",
        )
        if vae_model_name:
            vae = self._load_diffusers_sub_module(
                AutoencoderKLQwenImage21, weight_dtypes.vae, weight_dtypes.train_dtype, vae_model_name,
            )
        else:
            vae = self._load_diffusers_sub_module(
                AutoencoderKLQwenImage21, weight_dtypes.vae, weight_dtypes.train_dtype, base_model_name, "vae",
            )

        if transformer_model_name:
            transformer = QwenImage21Transformer2DModel.from_single_file(
                transformer_model_name,
                config=base_model_name,
                subfolder="transformer",
                torch_dtype=torch.bfloat16 if weight_dtypes.transformer.torch_dtype() is None else weight_dtypes.transformer.torch_dtype(),
                quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16)
                if weight_dtypes.transformer.is_gguf() else None,
            )
            transformer = self._convert_diffusers_sub_module_to_dtype(
                transformer, weight_dtypes.transformer, weight_dtypes.train_dtype, quantization,
            )
        else:
            transformer = self._load_diffusers_sub_module(
                QwenImage21Transformer2DModel,
                weight_dtypes.transformer,
                weight_dtypes.train_dtype,
                base_model_name,
                "transformer",
                quantization,
            )

        model.model_type = model_type
        model.processor = processor
        model.noise_scheduler = noise_scheduler
        model.text_encoder = text_encoder
        model.vae = vae
        model.transformer = transformer
        model.initialize_prompt_metadata()

    def __load_internal(self, *args):
        base_model_name = args[3]
        if not os.path.isfile(os.path.join(base_model_name, "meta.json")):
            raise Exception("not an internal model")
        self.__load_diffusers(*args)

    def load(
            self,
            model: QwenImage21Model,
            model_type: ModelType,
            model_names: ModelNames,
            weight_dtypes: ModelWeightDtypes,
            quantization: QuantizationConfig,
    ):
        args = (
            model, model_type, weight_dtypes, model_names.base_model,
            model_names.transformer_model, model_names.vae_model, quantization,
        )
        stacktraces = []
        for loader in (self.__load_internal, self.__load_diffusers):
            try:
                loader(*args)
                return
            except Exception:  # noqa: PERF203
                stacktraces.append(traceback.format_exc())
        for stacktrace in stacktraces:
            print(stacktrace)
        raise Exception("could not load model: " + model_names.base_model)
