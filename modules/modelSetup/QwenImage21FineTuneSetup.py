from modules.model.QwenImage21Model import QwenImage21Model
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.modelSetup.BaseQwenImage21Setup import BaseQwenImage21Setup
from modules.util import factory
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.ModuleFilter import ModuleFilter
from modules.util.NamedParameterGroup import NamedParameterGroupCollection
from modules.util.optimizer_util import init_model_parameters
from modules.util.TrainProgress import TrainProgress


@factory.register(BaseModelSetup, ModelType.QWEN_IMAGE_21, TrainingMethod.FINE_TUNE)
class QwenImage21FineTuneSetup(BaseQwenImage21Setup):
    def create_parameters(self, model, config):
        groups = NamedParameterGroupCollection()
        self._create_model_part_parameters(groups, "text_encoder", model.text_encoder, config.text_encoder)
        self._create_model_part_parameters(
            groups, "transformer", model.transformer, config.transformer,
            freeze=ModuleFilter.create(config), debug=config.debug_mode,
        )
        return groups

    def __setup_requires_grad(self, model, config):
        self._setup_model_part_requires_grad("text_encoder", model.text_encoder, config.text_encoder, model.train_progress)
        self._setup_model_part_requires_grad("transformer", model.transformer, config.transformer, model.train_progress)
        model.vae.requires_grad_(False)

    def setup_model(self, model: QwenImage21Model, config: TrainConfig):
        params = self.create_parameters(model, config)
        self.__setup_requires_grad(model, config)
        init_model_parameters(model, params, self.train_device)

    def setup_train_device(self, model, config):
        parts = ["transformer"]
        if not config.latent_caching:
            parts.extend(["text_encoder", "vae"])
        model.materialize_only(*parts)
        model.text_encoder.train(config.text_encoder.train)
        model.vae.eval()
        model.transformer.train(config.transformer.train)

    def after_optimizer_step(self, model, config, train_progress: TrainProgress):
        self.__setup_requires_grad(model, config)
