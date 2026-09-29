from modules.model.QwenImage21Model import QwenImage21Model
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.modelSetup.BaseQwenImage21Setup import BaseQwenImage21Setup
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util import factory
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.NamedParameterGroup import NamedParameterGroupCollection
from modules.util.optimizer_util import init_model_parameters
from modules.util.torch_util import state_dict_has_prefix
from modules.util.TrainProgress import TrainProgress


@factory.register(BaseModelSetup, ModelType.QWEN_IMAGE_21, TrainingMethod.LORA)
class QwenImage21LoRASetup(BaseQwenImage21Setup):
    def create_parameters(self, model, config):
        groups = NamedParameterGroupCollection()
        self._create_model_part_parameters(groups, "text_encoder", model.text_encoder_lora, config.text_encoder)
        self._create_model_part_parameters(groups, "transformer", model.transformer_lora, config.transformer)
        return groups

    def __setup_requires_grad(self, model, config):
        model.text_encoder.requires_grad_(False)
        model.transformer.requires_grad_(False)
        model.vae.requires_grad_(False)
        self._setup_model_part_requires_grad("text_encoder", model.text_encoder_lora, config.text_encoder, model.train_progress)
        self._setup_model_part_requires_grad("transformer", model.transformer_lora, config.transformer, model.train_progress)

    def setup_model(self, model: QwenImage21Model, config: TrainConfig):
        create_te = config.text_encoder.train or state_dict_has_prefix(model.lora_state_dict, "text_encoder")
        model.text_encoder_lora = LoRAModuleWrapper(model.text_encoder, "text_encoder", config) if create_te else None
        model.transformer_lora = LoRAModuleWrapper(
            model.transformer, "transformer", config, config.layer_filter.split(","),
        )
        if model.lora_state_dict:
            if model.text_encoder_lora is not None:
                model.text_encoder_lora.load_state_dict(model.lora_state_dict)
            model.transformer_lora.load_state_dict(model.lora_state_dict)
            model.lora_state_dict = None
        for wrapper in (model.text_encoder_lora, model.transformer_lora):
            if wrapper is not None:
                wrapper.set_dropout(config.dropout_probability)
                wrapper.to(dtype=config.lora_weight_dtype.torch_dtype())
                wrapper.hook_to_module()
        params = self.create_parameters(model, config)
        self.__setup_requires_grad(model, config)
        init_model_parameters(model, params, self.train_device)

    def setup_train_device(self, model, config):
        parts = ["transformer"]
        if not config.latent_caching:
            parts.extend(["text_encoder", "vae"])
        model.materialize_only(*parts)
        model.text_encoder.eval()
        model.vae.eval()
        model.transformer.train(config.transformer.train)

    def after_optimizer_step(self, model, config, train_progress: TrainProgress):
        self.__setup_requires_grad(model, config)
